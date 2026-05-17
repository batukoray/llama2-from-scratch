from typing import List
import math
import random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Math.Tensor import Tensor
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.RMSNorm import RMSNorm
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter


class Llama2(ComputationalGraph):
    """
    LLaMA 2 decoder-only model implementation.
    """

    __parameter: Llama2Parameter

    __input_node: ComputationalNode
    __random_generator: random.Random

    def __init__(self, parameter: Llama2Parameter):
        """
        Creates a new LLaMA 2 model with the given parameter object.
        """
        super().__init__(parameter)
        self.__parameter = parameter
        self.__random_generator = random.Random(self.__parameter.getSeed())

    def createOneHotVectors(self, token_ids: List[int]) -> Tensor:
        if len(self.input_nodes) == 0:
            raise ValueError("Input node must be created before calling createInputTensor.")

        values = []
        vocabulary_length = self.__parameter.getVocabularyLength()

        # For each token to convert to one hot encoding
        for token_id in token_ids:
            if token_id < 0 or token_id >= vocabulary_length:
                raise ValueError("Token id is out of vocabulary range.")

            # create the one hot array (flat):
            for i in range(vocabulary_length):
                if i == token_id:
                    values.append(1.0)
                else:
                    values.append(0.0)

        # tensor consisting of all one hot encodings
        return Tensor(values, (len(token_ids), vocabulary_length))

    def setInput(self, token_ids: List[int]) -> None:
        """
        Set the input to the model to the given token ids.
        """
        one_hot_tensor = self.createOneHotVectors(token_ids)
        self.__input_node.setValue(one_hot_tensor)

    def __createWeightNode(self, input_dimension: int, output_dimension: int) -> MultiplicationNode:
        """
        Creates a learnable matrix node with the given shape.
        """
        return MultiplicationNode(
            value=Tensor(
                self.__parameter.initializeWeights(
                    input_dimension,
                    output_dimension,
                    self.__random_generator
                ),
                (input_dimension, output_dimension)
            ),
            learnable=True,
            is_biased=False
        )

    def decoderBlock(self, current: ComputationalNode) -> ComputationalNode:
        """
        Builds one LLaMA 2 decoder block:
        (input)
        1. RMSNorm
        2. Causal self-attention with RoPE
        3. Residual (Add)
        4. RMSNorm
        5. SwiGLU feed-forward (?)
        6. Residual (Add)
        (output)
        """
        embedding_dimension = self.__parameter.getEmbeddingDimension()
        attention_head_count = self.__parameter.getAttentionHeadCount()
        head_dimension = embedding_dimension // attention_head_count
        epsilon = self.__parameter.getEpsilon()

        # 1. RMSNorm
        # output: (sequence_length, embedding_dimension)
        attention_input = self.addEdge(current, RMSNorm(embedding_dimension, epsilon))

        # 2. Causal self-attention with RoPE
        # <editor-fold desc="Causal self-attention...">
        attention_heads = []
        for _ in range(attention_head_count):
            # (embedding_dimension, head_dimension)
            wq = self.__createWeightNode(embedding_dimension, head_dimension)
            wk = self.__createWeightNode(embedding_dimension, head_dimension)
            wv = self.__createWeightNode(embedding_dimension, head_dimension)

            # (sequence_length, head_dimension)
            q = self.addEdge(attention_input, wq)
            k = self.addEdge(attention_input, wk)
            v = self.addEdge(attention_input, wv)

            # TODO: wire the base parameter of RoPE to a user-accessible place
            # Apply RoPE to Q and K (V isn't rotated in LLaMA 2)
            q_rope = self.addEdge(q, RotaryPositionEmbedding())
            k_rope = self.addEdge(k, RotaryPositionEmbedding())

            # get K^T for S = QK^T (head_dimension, sequence_length)
            k_transpose = self.addEdge(k_rope, Transpose())

            # S (raw attention score matrix): (sequence_length, sequence_length)
            S = self.addEdge(q_rope, k_transpose)

            # S_scaled = S / sqrt(d_k)
            S_scaled = self.addEdge(S, MultiplyByConstant(1.0 / math.sqrt(head_dimension)))

            # still (sequence_length, sequence_length)
            masked_scores = self.addEdge(S_scaled, Mask())
            attention_weights = self.addEdge(masked_scores, Softmax())

            # output: (sequence_length, head_dimension)
            attention_head = self.addEdge(attention_weights, v)
            attention_heads.append(attention_head)

        # (sequence_length, attention_head_count * head_dimension) = (sequence_length, embedding_dimension)
        concatenated_attention = self.concatEdges(attention_heads, 1)

        wo = self.__createWeightNode(embedding_dimension, embedding_dimension)
        attention_output = self.addEdge(concatenated_attention, wo) # (sequence_length, embedding_dimension)
        # </editor-fold>

        raise ValueError("not implemented yet")

    def buildGraph(self) -> None:
        """
        Builds the decoder-only forward path from token ids to embedding,
        N decoder blocks with RMSNorm, masked self-attention with RoPE, residuals,
        SwiGLU feed-forward, final RMSNorm, lm_head, and Softmax.
        """
        # used when creating E and lm_head
        vocab_length = self.__parameter.getVocabularyLength()
        embedding_dimension = self.__parameter.getEmbeddingDimension()

        # used when creating RMSNorm nodes
        epsilon = self.__parameter.getEpsilon()

        # Create the input node.
        input_node = MultiplicationNode(learnable=False, is_biased=False)
        self.input_nodes.append(input_node)
        self.__input_node = input_node

        # Create embedding matrix E
        # (one hot vector * E) = embeddings
        embedding_node = self.__createWeightNode(
            input_dimension=vocab_length,
            output_dimension=embedding_dimension,
        )
        current = self.addEdge(input_node, embedding_node)

        # decoder blocks
        for _ in range(self.__parameter.getDecoderLayerCount()):
            current = self.decoderBlock(current)

        # final RMSNorm
        current = self.addEdge(
            current,
            RMSNorm(embedding_dimension, epsilon)
        )

        # lm_head -> logits -> softmax -> output
        lm_head = self.__createWeightNode(embedding_dimension, vocab_length)
        logits = self.addEdge(current, lm_head)
        self.output_node = self.addEdge(logits, Softmax())

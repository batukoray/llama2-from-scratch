from typing import List
import random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Math.Tensor import Tensor
from SequenceProcessing.Functions.RMSNorm import RMSNorm
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Functions.SiLU import SiLU
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter


class Llama2(ComputationalGraph):
    """
    LLaMA 2 decoder-only model implementation.
    """

    __parameter: Llama2Parameter

    __input_node: ComputationalNode

    def __init__(self, parameter: Llama2Parameter):
        """
        Creates a new LLaMA 2 model with the given parameter object.
        """
        super().__init__(parameter)
        self.__parameter = parameter

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

    def __createWeightNode(self,
                           input_dimension: int,
                           output_dimension: int,
                           random_generator: random.Random) -> MultiplicationNode:
        """
        Creates a learnable matrix node with the given shape.
        """
        return MultiplicationNode(
            value=Tensor(
                self.__parameter.initializeWeights(
                    input_dimension,
                    output_dimension,
                    random_generator
                ),
                (input_dimension, output_dimension)
            ),
            learnable=True,
            is_biased=False
        )

    def decoderBlock(self,
                     current: ComputationalNode,
                     random_generator: random.Random) -> ComputationalNode:
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

        raise ValueError("Not implemented yet.")

    def buildGraph(self) -> None:
        """
        Builds the decoder-only forward path from token ids to embedding,
        N decoder blocks with RMSNorm, masked self-attention with RoPE, residuals,
        SwiGLU feed-forward, final RMSNorm, lm_head, and Softmax.
        """
        # used when creating E and lm_head
        random_generator = random.Random(self.__parameter.getSeed())
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
            random_generator=random_generator
        )
        current = self.addEdge(input_node, embedding_node)

        # decoder blocks
        for _ in range(self.__parameter.getDecoderLayerCount()):
            current = self.decoderBlock(current, random_generator)

        # final RMSNorm
        current = self.addEdge(
            current,
            RMSNorm(embedding_dimension, epsilon)
        )

        # lm_head -> logits -> softmax -> output
        lm_head = self.__createWeightNode(embedding_dimension, vocab_length, random_generator)
        logits = self.addEdge(current, lm_head)
        self.output_node = self.addEdge(logits, Softmax())

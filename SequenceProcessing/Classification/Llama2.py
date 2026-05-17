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
        self.input_nodes[0].setValue(one_hot_tensor)

    def buildGraph(self) -> None:
        """
        Builds the decoder-only forward path from token ids to embedding, N decoder blocks with RMSNorm, masked self-attention with RoPE, residuals, SwiGLU feed-forward, final RMSNorm, lm_head, and Softmax.
        """
        # Create the input node.
        input_node = MultiplicationNode(learnable=False, is_biased=True)
        self.input_nodes.append(input_node)

        # Create embedding matrix E
        vocab_length = self.__parameter.getVocabularyLength()
        embedding_dimension = self.__parameter.getEmbeddingDimension()
        random_generator = random.Random(self.__parameter.getSeed())
        embedding_matrix = Tensor(
            self.__parameter.initializeWeights(
                vocab_length,
                embedding_dimension,
                random_generator
            ),
            (vocab_length, embedding_dimension)
        )

        # Wrap the embedding matrix inside a MultiplicationNode
        # because (one hot vector * E) = embeddings
        # This node is used to convert input tokens to embeddings
        embedding_node = MultiplicationNode(embedding_matrix, (vocab_length, embedding_dimension))
        self.addEdge(input_node, embedding_node)

        # Decoder blocks, final RMSNorm, lm_head projection, and Softmax go here.

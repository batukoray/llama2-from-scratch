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

    def createInputTensor(self, token_ids: List[int]) -> None:
        """
        Creates the embedded input tensor from token ids.
        """
        if len(self.input_nodes) == 0:
            raise ValueError("Input node must be created before calling createInputTensor.")

        values = []
        vocabulary_length = self.__parameter.getVocabularyLength()
        embedding_dimension = self.__parameter.getEmbeddingDimension()

        for token_id in token_ids:
            if token_id < 0 or token_id >= vocabulary_length:
                raise ValueError("Token id is out of vocabulary range.")

            for i in range(vocabulary_length):
                if i == token_id:
                    values.append(1.0)
                else:
                    values.append(0.0)

        one_hot_tensor = Tensor(values, (len(token_ids), vocabulary_length))

        random_generator = random.Random(self.__parameter.getSeed())
        embedding_weight = MultiplicationNode(
            Tensor(
                self.__parameter.initializeWeights(
                    vocabulary_length,
                    embedding_dimension,
                    random_generator
                ),
                (vocabulary_length, embedding_dimension)
            )
        )

        self.input_nodes[0].setValue(one_hot_tensor.multiply(embedding_weight.getValue()))

    def buildGraph(self) -> None:
        """
        Builds the decoder-only forward path from token ids to embedding, N decoder blocks with RMSNorm, masked self-attention with RoPE, residuals, SwiGLU feed-forward, final RMSNorm, lm_head, and Softmax.
        """
        input_node = MultiplicationNode(False, True)
        self.input_nodes.append(input_node)

        # Decoder blocks, final RMSNorm, lm_head projection, and Softmax go here.

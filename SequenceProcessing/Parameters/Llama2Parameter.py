from typing import Optional

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.AdamW import AdamW
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class Llama2Parameter(NeuralNetworkParameter):
    """
    Holds the training and architecture settings for a LLaMA 2 style model.

    The existing transformer parameter class in this repository is shaped around
    an encoder-decoder transformer. A decoder-only language model has a simpler
    structure, so it helps to keep its settings in a separate class. This class
    stores the dimensions that define a LLaMA 2 style model and also inherits
    the common training settings from ``NeuralNetworkParameter``.
    """

    __vocabulary_length: int
    __embedding_dimension: int
    __decoder_layer_count: int
    __attention_head_count: int
    __key_value_head_count: int
    __context_length: int
    __feed_forward_dimension: int
    __epsilon: float

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 vocabulary_length: int,
                 embedding_dimension: int,
                 decoder_layer_count: int,
                 attention_head_count: int,
                 key_value_head_count: Optional[int],
                 context_length: int,
                 feed_forward_dimension: int,
                 epsilon: float,
                 dropout: float = 0.0,
                 batch_size: int = 1):
        """
        Creates a new parameter object for a decoder-only language model.

        :param seed: Random seed used during initialization and training.
        :param epoch: Number of training epochs.
        :param optimizer: Optimization method used by the model.
        :param initialization: Weight initialization strategy.
        :param loss: Loss function used during training.
        :param vocabulary_length: Number of tokens in the vocabulary.
        :param embedding_dimension: Width of the token representation.
        :param decoder_layer_count: Number of decoder blocks in the model.
        :param attention_head_count: Number of query heads used in attention.
        :param key_value_head_count: Number of key and value heads. If this is
                                     not given, the model uses the same number
                                     as the attention head count.
        :param context_length: Maximum number of tokens processed at once.
        :param feed_forward_dimension: Hidden size of the SwiGLU feed-forward block.
        :param epsilon: Small numeric constant used in RMS normalization.
        :param dropout: Dropout ratio inherited from the base parameter class.
        :param batch_size: Batch size inherited from the base parameter class.
        """
        super().__init__(
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            initialization=initialization,
            loss_function=loss,
            dropout=dropout,
            batch_size=batch_size
        )

        if key_value_head_count is None:
            key_value_head_count = attention_head_count

        self.__validateConfiguration(
            vocabulary_length,
            embedding_dimension,
            decoder_layer_count,
            attention_head_count,
            key_value_head_count,
            context_length,
            feed_forward_dimension,
            epsilon
        )

        self.__vocabulary_length = vocabulary_length
        self.__embedding_dimension = embedding_dimension
        self.__decoder_layer_count = decoder_layer_count
        self.__attention_head_count = attention_head_count
        self.__key_value_head_count = key_value_head_count
        self.__context_length = context_length
        self.__feed_forward_dimension = feed_forward_dimension
        self.__epsilon = epsilon

    def __validateConfiguration(self,
                                vocabulary_length: int,
                                embedding_dimension: int,
                                decoder_layer_count: int,
                                attention_head_count: int,
                                key_value_head_count: int,
                                context_length: int,
                                feed_forward_dimension: int,
                                epsilon: float) -> None:
        """
        Checks whether the provided model settings are internally consistent.

        :param vocabulary_length: Number of vocabulary entries.
        :param embedding_dimension: Width of the token representation.
        :param decoder_layer_count: Number of decoder blocks.
        :param attention_head_count: Number of attention heads.
        :param key_value_head_count: Number of key and value heads.
        :param context_length: Maximum sequence length.
        :param feed_forward_dimension: Width of the feed-forward block.
        :param epsilon: Small numeric constant used in normalization.
        """
        if vocabulary_length <= 0:
            raise ValueError("vocabulary_length must be positive.")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")
        if decoder_layer_count <= 0:
            raise ValueError("decoder_layer_count must be positive.")
        if attention_head_count <= 0:
            raise ValueError("attention_head_count must be positive.")
        if key_value_head_count <= 0:
            raise ValueError("key_value_head_count must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if feed_forward_dimension <= 0:
            raise ValueError("feed_forward_dimension must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if embedding_dimension % attention_head_count != 0:
            raise ValueError("embedding_dimension must be divisible by attention_head_count.")
        if attention_head_count % key_value_head_count != 0:
            raise ValueError("attention_head_count must be divisible by key_value_head_count.")

    @classmethod
    def tinyLlama2(cls,
                   seed: int = 1,
                   epoch: int = 10,
                   optimizer: Optional[Optimizer] = None,
                   initialization: Optional[Initialization] = None,
                   loss: Optional[Function] = None) -> "Llama2Parameter":
        """
        Builds a very small configuration for tests and toy experiments.

        The values here are intentionally modest so that later graph-building
        and training code can be debugged on a local machine without needing
        large amounts of memory.

        :param seed: Random seed used during initialization and training.
        :param epoch: Number of training epochs.
        :param optimizer: Optional optimizer. A default AdamW optimizer is used if none is given.
        :param initialization: Optional initialization method.
        :param loss: Optional loss function.
        :return: A compact LLaMA 2 style parameter object.
        """
        if optimizer is None:
            optimizer = AdamW(0.001, 0.99, 0.9, 0.999, 1e-8, 0.0)
        if initialization is None:
            initialization = RandomInitialization()
        if loss is None:
            loss = CrossEntropyLoss()

        return cls(
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            initialization=initialization,
            loss=loss,
            vocabulary_length=256,
            embedding_dimension=64,
            decoder_layer_count=2,
            attention_head_count=4,
            key_value_head_count=4,
            context_length=32,
            feed_forward_dimension=256,
            epsilon=1e-6
        )

    @classmethod
    def llama2_7B(cls,
                  seed: int = 1,
                  epoch: int = 1,
                  optimizer: Optional[Optimizer] = None,
                  initialization: Optional[Initialization] = None,
                  loss: Optional[Function] = None) -> "Llama2Parameter":
        """
        Builds a reference configuration that matches the published 7B layout.

        This helper is useful when code needs the real architectural numbers,
        but it should not be mistaken for a practical local training setup.

        :param seed: Random seed used during initialization and training.
        :param epoch: Number of training epochs.
        :param optimizer: Optional optimizer. A default AdamW optimizer is used if none is given.
        :param initialization: Optional initialization method.
        :param loss: Optional loss function.
        :return: A parameter object that mirrors the 7B model dimensions.
        """
        if optimizer is None:
            optimizer = AdamW(0.001, 0.99, 0.9, 0.999, 1e-8, 0.0)
        if initialization is None:
            initialization = RandomInitialization()
        if loss is None:
            loss = CrossEntropyLoss()

        return cls(
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            initialization=initialization,
            loss=loss,
            vocabulary_length=32000,
            embedding_dimension=4096,
            decoder_layer_count=32,
            attention_head_count=32,
            key_value_head_count=32,
            context_length=4096,
            feed_forward_dimension=11008,
            epsilon=1e-5
        )

    def getVocabularyLength(self) -> int:
        """
        Returns the size of the token vocabulary.

        :return: Vocabulary size.
        """
        return self.__vocabulary_length

    def getEmbeddingDimension(self) -> int:
        """
        Returns the width of each token representation.

        :return: Embedding dimension.
        """
        return self.__embedding_dimension

    def getDecoderLayerCount(self) -> int:
        """
        Returns the number of decoder blocks.

        :return: Decoder layer count.
        """
        return self.__decoder_layer_count

    def getAttentionHeadCount(self) -> int:
        """
        Returns the number of attention heads.

        :return: Attention head count.
        """
        return self.__attention_head_count

    def getKeyValueHeadCount(self) -> int:
        """
        Returns the number of key and value heads.

        :return: Key and value head count.
        """
        return self.__key_value_head_count

    def getContextLength(self) -> int:
        """
        Returns the maximum supported sequence length.

        :return: Context length.
        """
        return self.__context_length

    def getFeedForwardDimension(self) -> int:
        """
        Returns the hidden size used inside the feed-forward block.

        :return: Feed-forward dimension.
        """
        return self.__feed_forward_dimension

    def getEpsilon(self) -> float:
        """
        Returns the numeric constant used by RMS normalization.

        :return: Epsilon value.
        """
        return self.__epsilon

    def getHeadDimension(self) -> int:
        """
        Returns the width of each attention head.

        :return: Dimension per attention head.
        """
        return self.__embedding_dimension // self.__attention_head_count

    def usesGroupedQueryAttention(self) -> bool:
        """
        Tells whether the configuration uses grouped-query attention.

        When the number of key and value heads is smaller than the number of
        query heads, the model is using grouped-query attention.

        :return: True if grouped-query attention is active, False otherwise.
        """
        return self.__key_value_head_count != self.__attention_head_count

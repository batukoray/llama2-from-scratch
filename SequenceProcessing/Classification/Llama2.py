from typing import List, Optional
import math
import random

from Classification.Performance.ClassificationPerformance import ClassificationPerformance
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Math.Tensor import Tensor
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.RMSNorm import RMSNorm
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Functions.SiLU import SiLU
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
        Also checks for context length.
        """
        if len(self.input_nodes) == 0:
            raise ValueError("Input node must be created before calling setInput.")
        if len(token_ids) > self.__parameter.getContextLength():
            raise ValueError("Token id sequence exceeds the configured context length.")

        one_hot_tensor = self.createOneHotVectors(token_ids)
        self.__input_node.setValue(one_hot_tensor)

        if len(self.input_nodes) > 1:
            self.input_nodes[1].setValue(
                Tensor(
                    [0.0] * (len(token_ids) * self.__parameter.getVocabularyLength()),
                    (len(token_ids), self.__parameter.getVocabularyLength())
                )
            )

    def setLabels(self, token_ids: List[int]) -> None:
        self.input_nodes[1].setValue(self.createOneHotVectors(token_ids))

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

    def __addRMSNorm(self, current: ComputationalNode, dimension: int) -> ComputationalNode:
        """
        Adds RMS normalization followed by a learnable gamma scale.
        """
        normalized = self.addEdge(current, RMSNorm(epsilon=self.__parameter.getEpsilon()))

        # Shape (1, dimension) broadcasts across sequence rows and lets the
        # optimizer reduce gamma gradients back to the parameter shape.
        gamma = MultiplicationNode(
            value=Tensor([1.0] * dimension, shape=(1, dimension)),
            learnable=True,
            is_biased=False,
            is_hadamard=True
        )
        return self.addEdge(normalized, gamma)

    def decoderBlock(self, current: ComputationalNode) -> ComputationalNode:
        """
        Builds one LLaMA 2 decoder block:
        (input)
        1. RMSNorm
        2. Causal self-attention with RoPE
        3. Residual (Add)
        4. RMSNorm
        5. SwiGLU feed-forward
        6. Residual (Add)
        (output)
        """
        embedding_dimension = self.__parameter.getEmbeddingDimension()
        attention_head_count = self.__parameter.getAttentionHeadCount()
        key_value_head_count = self.__parameter.getKeyValueHeadCount()
        head_dimension = embedding_dimension // attention_head_count
        feed_forward_dimension = self.__parameter.getFeedForwardDimension()

        if head_dimension % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE.")

        # 1. RMSNorm
        # output: (sequence_length, embedding_dimension)
        attention_input = self.__addRMSNorm(current, embedding_dimension)

        # 2. Causal self-attention with RoPE
        # <editor-fold desc="Causal self-attention...">
        key_heads = []
        value_heads = []
        for _ in range(key_value_head_count):
            # (embedding_dimension, head_dimension)
            wk = self.__createWeightNode(embedding_dimension, head_dimension)
            wv = self.__createWeightNode(embedding_dimension, head_dimension)

            # (sequence_length, head_dimension)
            k = self.addEdge(attention_input, wk)
            v = self.addEdge(attention_input, wv)

            # TODO: wire the base parameter of RoPE to a user-accessible place
            # Apply RoPE to K (V isn't rotated in LLaMA 2)
            key_heads.append(self.addEdge(k, RotaryPositionEmbedding()))
            value_heads.append(v)

        attention_heads = []
        for head_index in range(attention_head_count):
            # (embedding_dimension, head_dimension)
            wq = self.__createWeightNode(embedding_dimension, head_dimension)

            # (sequence_length, head_dimension)
            q = self.addEdge(attention_input, wq)

            # Apply RoPE to Q.
            q_rope = self.addEdge(q, RotaryPositionEmbedding())

            # for grouped-query attention
            key_value_index = head_index * key_value_head_count // attention_head_count

            # get K^T for S = QK^T (head_dimension, sequence_length)
            k_transpose = self.addEdge(key_heads[key_value_index], Transpose())

            # S (raw attention score matrix): (sequence_length, sequence_length)
            S = self.addEdge(q_rope, k_transpose, False, False)

            # S_scaled = S / sqrt(d_k)
            S_scaled = self.addEdge(S, MultiplyByConstant(1.0 / math.sqrt(head_dimension)))

            # still (sequence_length, sequence_length)
            masked_scores = self.addEdge(S_scaled, Mask())
            attention_weights = self.addEdge(masked_scores, Softmax())

            # output: (sequence_length, head_dimension)
            attention_head = self.addEdge(attention_weights, value_heads[key_value_index])
            attention_heads.append(attention_head)

        # (sequence_length, attention_head_count * head_dimension) = (sequence_length, embedding_dimension)
        concatenated_attention = self.concatEdges(attention_heads, 1)

        wo = self.__createWeightNode(embedding_dimension, embedding_dimension)
        attention_output = self.addEdge(concatenated_attention, wo) # (sequence_length, embedding_dimension)
        # </editor-fold>

        # 3. Residual: add original block input + attention output.
        attention_residual = self.addAdditionEdge(current, attention_output, False)

        # 4. RMSNorm
        feed_forward_input = self.__addRMSNorm(attention_residual, embedding_dimension)

        # 5. SwiGLU feed-forward network.
        # SwiGLU(x) = SiLU(xW1) hadamard (xW2)
        # FFN(x) = W3(SwiGLU(x))
        # W1, W2: embedding_dimension -> feed_forward_dimension
        # W3: feed_forward_dimension -> embedding_dimension

        w1 = self.__createWeightNode(embedding_dimension, feed_forward_dimension)
        w2 = self.__createWeightNode(embedding_dimension, feed_forward_dimension)
        w3 = self.__createWeightNode(feed_forward_dimension, embedding_dimension)

        # gate = SiLU(xW1)
        gate = self.addEdge(feed_forward_input, w1)
        gate = self.addEdge(gate, SiLU())

        # up = xW2
        up = self.addEdge(feed_forward_input, w2)

        # swiglu = SiLU(xW1) * (xW2)
        swiglu = self.addEdge(gate, up, False, True)

        # feed_forward_output = swiglu W3
        feed_forward_output = self.addEdge(swiglu, w3)

        # 6. Residual: add attention residual + feed-forward output.
        return self.addAdditionEdge(attention_residual, feed_forward_output, False)

    def buildGraph(self) -> None:
        """
        Builds the decoder-only forward path from token ids to embedding,
        N decoder blocks with RMSNorm, masked self-attention with RoPE, residuals,
        SwiGLU feed-forward, final RMSNorm, lm_head, and Softmax.
        """
        # used when creating E and lm_head
        vocab_length = self.__parameter.getVocabularyLength()
        embedding_dimension = self.__parameter.getEmbeddingDimension()

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
        current = self.__addRMSNorm(current, embedding_dimension)

        # lm_head -> logits -> softmax -> output
        lm_head = self.__createWeightNode(embedding_dimension, vocab_length)
        logits = self.addEdge(current, lm_head)
        self.output_node = self.addEdge(logits, Softmax())

        # for training/testing:
        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, self.__parameter.getLossFunction(), False)

    def __ensureGraph(self) -> None:
        """
        Makes sure the graph exists.
        """
        if len(self.input_nodes) == 0 or self.output_node is None:
            self.buildGraph()

        if getattr(self, "_ComputationalGraph__leaf_nodes", None) is None:
            self._ComputationalGraph__leaf_nodes = self._ComputationalGraph__findLeafNodes()

    def predictNextToken(self, token_ids: List[int]) -> int:
        """
        Predicts the next token after the given token ids.
        """
        self.__ensureGraph()

        if len(token_ids) == 0:
            raise ValueError("At least one token is required.")

        # trim to context length
        if len(token_ids) > self.__parameter.getContextLength():
            token_ids = token_ids[-self.__parameter.getContextLength():]

        self.setInput(token_ids)
        y_pred = self.predict()

        return int(y_pred[-1])
    def generateGreedy(self,
                       token_ids: List[int],
                       max_new_tokens: int,
                       end_token_id: Optional[int] = None) -> List[int]:
        """
        Generates tokens greedily from the current model.
        """
        generated_token_ids = list(token_ids)

        for _ in range(max_new_tokens):
            next_token_id = self.predictNextToken(generated_token_ids)
            generated_token_ids.append(next_token_id)

            if end_token_id is not None and next_token_id == end_token_id:
                break

        return generated_token_ids


    @staticmethod
    def __tensorToTokenIds(instance: Tensor) -> List[int]:
        """
        Converts a 1D token-id tensor into token ids.
        """
        shape = instance.getShape()
        if len(shape) != 1:
            raise ValueError("Llama2 expects each instance to be a 1D token-id Tensor.")

        token_ids = []
        for i in range(shape[0]):
            token_value = instance.getValue((i,))
            token_id = int(token_value)
            token_ids.append(token_id)

        return token_ids

    def __createInputAndLabels(self, token_ids: List[int]) -> tuple[List[int], List[int]]:
        """
        Sets model input to all tokens except the last one and returns next-token labels.
        e.g., for input [1,2,3,4]
        returns ([1,2,3], [2,3,4])
        for input_token_ids and labels respectively
        """
        if len(token_ids) < 2:
            return [],[]

        maximum_length = self.__parameter.getContextLength() + 1
        if len(token_ids) > maximum_length:
            token_ids = token_ids[-maximum_length:]

        input_token_ids = token_ids[:-1]
        class_labels = token_ids[1:]

        return input_token_ids, class_labels

    def train(self, train_set: List[Tensor]) -> None:
        """
        Trains the model as a next-token language model.

        :param train_set: Training sequences represented as 1D token-id tensors.
        """
        self.__ensureGraph()
        random_generator = random.Random(self.__parameter.getSeed())

        # Epoch
        for _ in range(self.__parameter.getEpoch()):
            # Shuffle
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            # Step
            for tensor in train_set:
                token_ids = self.__tensorToTokenIds(tensor)
                input_token_ids, labels = self.__createInputAndLabels(token_ids)
                if len(labels) == 0:
                    continue

                self.setInput(input_token_ids)
                self.setLabels(labels)
                self.forwardCalculation()
                self.backpropagation()

            self.__parameter.getOptimizer().setLearningRate()

    def test(self, test_set: List[Tensor]):
        """
        Tests next-token prediction accuracy.

        :param test_set: Test sequences represented as 1D token-id tensors.
        :return: Classification performance.
        """
        self.__ensureGraph()
        correct = 0
        total = 0

        for tensor in test_set:
            token_ids = self.__tensorToTokenIds(tensor)
            input_token_ids, labels = self.__createInputAndLabels(token_ids)
            if len(labels) == 0:
                continue

            self.setInput(input_token_ids)

            predicted = self.predict()

            for i in range(len(labels)):
                if int(predicted[i]) == labels[i]:
                    correct += 1
                total += 1

        if total == 0:
            return ClassificationPerformance(0.0)

        return ClassificationPerformance((correct + 0.0) / total)

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts predicted token ids from the output node.
        Direct copy from Transformer.py

        :param output_node: Output node.
        :return: Predicted token ids.
        """
        class_labels = []
        value = output_node.getValue()

        for i in range(value.getShape()[0]):
            max_val = float("-inf")
            index = -1.0

            for j in range(value.getShape()[1]):
                current = value.getValue((i, j))
                if current > max_val:
                    max_val = current
                    index = float(j)

            class_labels.append(index)

        return class_labels

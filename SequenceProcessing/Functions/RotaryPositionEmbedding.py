import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RotaryPositionEmbedding(Function):
    """
    Applies RoPE to a 2D tensor interpreted as (position, feature).

    For position p, pair index i, and feature width d:
        theta(p, i, d) = p / base^(2i / d)

    The pair (x_{2i}, x_{2i+1}) is rotated as:
        y_{2i}   = x_{2i} * cos(theta) - x_{2i+1} * sin(theta)
        y_{2i+1} = x_{2i} * sin(theta) + x_{2i+1} * cos(theta)
    """

    __base: float

    def __init__(self, base: float = 10000.0):
        """
        Creates the RoPE operator.

        :param base: Frequency base in theta(p, i, d) = p / base^(2i / d).
        """
        self.__base = float(base)

    @staticmethod
    def __checkShape(tensor: Tensor) -> None:
        """
        Validates the tensor shape expected by RoPE.

        :param tensor: Tensor to validate. It must be 2D and have an even last
                       dimension so features can be rotated in pairs.
        """
        shape = tensor.getShape()

        if len(shape) != 2:
            raise ValueError("RotaryPositionEmbedding expects a 2D tensor.")
        if shape[1] % 2 != 0:
            raise ValueError("RotaryPositionEmbedding requires an even last dimension.")

    def __angle(self, position: int, i: int, d: int) -> float:
        """
        Computes the rotation angle theta for one position-feature pair.

        :param position: Token position p along the sequence axis.
        :param i: Pair index i, where features (2i, 2i + 1) are rotated together.
        :param d: Full feature width d of the last tensor dimension.
        :return: Rotation angle theta(p, i, d).
        """
        return position / math.pow(self.__base, (2.0 * i) / d)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies the forward RoPE rotation to each feature pair.

        :param tensor: Input tensor x with shape (sequence_length, feature_width).
        :return: Rotated tensor y with the same shape.
        """
        self.__checkShape(tensor)

        values = []
        shape = tensor.getShape()

        n = shape[0]
        d = shape[1]

        for position in range(n):
            for pair_index in range(d // 2):
                angle = self.__angle(position, pair_index, d)
                cos_value = math.cos(angle)
                sin_value = math.sin(angle)

                first = 2 * pair_index
                second = first + 1

                even_value = tensor.getValue((position, first))
                odd_value = tensor.getValue((position, second))

                values.append(even_value * cos_value - odd_value * sin_value)
                values.append(even_value * sin_value + odd_value * cos_value)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Applies the inverse rotation to the incoming gradient.

        If the forward pass uses rotation matrix R(theta), then the backward
        pass applies R(theta)^T to each gradient pair:
            g'_{2i}   = g_{2i} * cos(theta) + g_{2i+1} * sin(theta)
            g'_{2i+1} = -g_{2i} * sin(theta) + g_{2i+1} * cos(theta)

        :param value: Forward output tensor y. It is not used directly.
        :param backward: Incoming gradient dL/dy.
        :return: Outgoing gradient dL/dx after inverse rotation.
        """
        self.__checkShape(backward)

        values = []
        shape = backward.getShape()
        n = shape[0]
        d = shape[1]

        for position in range(n):
            for pair_index in range(d // 2):
                angle = self.__angle(position, pair_index, d)
                cos_value = math.cos(angle)
                sin_value = math.sin(angle)

                first = 2 * pair_index
                second = first + 1

                even_backward = backward.getValue((position, first))
                odd_backward = backward.getValue((position, second))

                values.append(even_backward * cos_value + odd_backward * sin_value)
                values.append(-even_backward * sin_value + odd_backward * cos_value)

        return Tensor(values, shape)

    def addEdge(self, input_nodes: List[ComputationalNode], is_biased: bool) -> ComputationalNode:
        """
        Attaches this operator to the computational graph.

        :param input_nodes: Source nodes that provide the tensor to rotate.
        :param is_biased: Whether the created graph edge is marked as biased.
        :return: Newly created function node representing RoPE.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RotaryPositionEmbedding(Function):

    __base: float

    def __init__(self, base: float = 10000.0):
        """
        Constructor for RotaryPositionEmbedding.

        :param base: Frequency base used by RoPE.
        """
        self.__base = float(base)

    @staticmethod
    def __checkShape(tensor: Tensor) -> None:
        """
        Validates that the tensor can be used with RoPE.
        """
        shape = tensor.getShape()

        if len(shape) != 2:
            raise ValueError("RotaryPositionEmbedding expects a 2D tensor.")
        if shape[1] % 2 != 0:
            raise ValueError("RotaryPositionEmbedding requires an even last dimension.")

    def __angle(self, position: int, i: int, d: int) -> float:
        """
        Calculates the rotation angle for one position and dimension pair.
        :param i: Pair index
        :param d: Dimension
        """
        return position / math.pow(self.__base, (2.0 * i) / d)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies RoPE in the forward pass.
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
        Applies the inverse rotation to the backward gradient.
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
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

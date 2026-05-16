import math
from typing import List, Optional

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SiLU(Function):
    """
    Applies the SiLU activation function to each tensor element.
    """

    __last_input: Optional[Tensor]

    def __init__(self):
        """
        Constructor for SiLU.
        """
        self.__last_input = None

    def __sigmoid(self, value: float) -> float:
        """
        Computes the sigmoid value for a single number.
        """
        return 1.0 / (1.0 + math.exp(-value))

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies SiLU to the input tensor and stores the input.
        """
        self.__last_input = tensor

        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_value = tensor.getValue((i, j))
                sigmoid_value = self.__sigmoid(current_value)
                values.append(current_value * sigmoid_value)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the SiLU derivative using the stored input tensor.
        """
        if self.__last_input is None:
            raise ValueError("SiLU derivative requires a previous calculate call.")

        values = []
        shape = self.__last_input.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_value = self.__last_input.getValue((i, j))
                sigmoid_value = self.__sigmoid(current_value)
                derivative_value = sigmoid_value + current_value * sigmoid_value * (1.0 - sigmoid_value)
                values.append(derivative_value)

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

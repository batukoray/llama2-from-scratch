import math
from typing import List, Optional

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SiLU(Function):
    """
    Applies the SiLU activation function element-wise.

    With sigma(x) = 1 / (1 + exp(-x)):
        SiLU(x) = x * sigma(x)
    """

    __last_input: Optional[Tensor]

    def __init__(self):
        """
        Creates the SiLU activation operator.
        """
        self.__last_input = None

    def __sigmoid(self, value: float) -> float:
        """
        Computes sigma(x) = 1 / (1 + exp(-x)).

        :param value: Scalar input x.
        :return: Sigmoid value sigma(x).
        """
        return 1.0 / (1.0 + math.exp(-value))

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies the forward SiLU map y = x * sigma(x).

        :param tensor: Input tensor x.
        :return: Output tensor y whose entries are x_ij * sigma(x_ij).
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
        Applies the SiLU backward rule using the stored forward input.

        For y = x * sigma(x), the local derivative is:
            dy/dx = sigma(x) + x * sigma(x) * (1 - sigma(x))
        and the returned gradient is:
            dL/dx = dL/dy hadamard dy/dx

        :param value: Forward output tensor y. It is not used directly because
                      the implementation reuses the stored input tensor.
        :param backward: Incoming gradient dL/dy.
        :return: Outgoing gradient dL/dx.
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
        Attaches this operator to the computational graph.

        :param input_nodes: Source nodes that provide the input tensor x.
        :param is_biased: Whether the created graph edge is marked as biased.
        :return: Newly created function node representing SiLU activation.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

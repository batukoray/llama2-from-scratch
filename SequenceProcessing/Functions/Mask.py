from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Mask(Function):
    """
    Applies a causal mask to a tensor.
    """

    def __init__(self):
        """
        Constructor for Mask.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies a causal mask to a square attention score matrix.

        For a score matrix S of shape (T, T), every entry above the main
        diagonal is set to negative infinity:

            S_masked[i, j] = S[i, j]   if j <= i
            S_masked[i, j] = -inf       if j > i

        When softmax is applied afterwards, -inf maps to exactly 0, so
        position i can only attend to positions 0 through i. This enforces
        the autoregressive constraint: no token may look at future tokens.

        :param tensor: Square attention score matrix S of shape (T, T).
        :return: Masked score matrix where all future positions are -inf.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                if j > i:
                    values.append(float("-inf"))
                else:
                    values.append(tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of the mask operation.

        Since the Java version multiplies the backward tensor by a tensor
        filled with ones, the backward tensor is preserved unchanged.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0)

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node
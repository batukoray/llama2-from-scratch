import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class StableSoftmax(Function):
    """
    Applies a numerically stable softmax over the last tensor dimension.
    """

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes softmax values after shifting each row by its maximum.
        """
        old_values = tensor.getData()
        shape = tensor.getShape()
        last_dim_size = shape[-1]

        new_data = []

        for i in range(0, len(old_values), last_dim_size):
            chunk = old_values[i:i + last_dim_size]
            maximum_value = max(chunk)
            exp_chunk = [math.exp(value - maximum_value) for value in chunk]
            chunk_sum = sum(exp_chunk)

            new_data.extend([exp_value / chunk_sum for exp_value in exp_chunk])

        return Tensor(new_data, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the softmax derivative from the output tensor.
        """
        old_values = tensor.getData()
        backward_values = backward.getData()
        shape = tensor.getShape()
        last_dim_size = shape[-1]

        new_values = []

        for i in range(0, len(old_values), last_dim_size):
            output_chunk = old_values[i:i + last_dim_size]
            backward_chunk = backward_values[i:i + last_dim_size]
            total = sum(output * grad for output, grad in zip(output_chunk, backward_chunk))

            new_values.extend([grad - total for grad in backward_chunk])

        return tensor.hadamardProduct(Tensor(new_values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.
        """
        new_node = FunctionNode(function=self, is_biased=is_biased)
        input_nodes[0].add(new_node)
        return new_node

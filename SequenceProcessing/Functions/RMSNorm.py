from typing import List, Optional

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RMSNorm(Function):
    """
    Applies RMSNorm with weight.
    """

    __epsilon: float
    __last_input: Optional[Tensor]

    def __init__(self, epsilon: float = 1e-6):
        """
        Constructor for RMSNorm.

        :param epsilon: Small constant added to the row-wise mean square.
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.__epsilon = float(epsilon)
        self.__last_input = None

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Performs row-wise RMS normalization over the last tensor dimension.

        :param tensor: Input tensor.
        :return: RMS-normalized tensor.
        """
        values = []
        shape = tensor.getShape()
        row_size = shape[-1]
        data = tensor.getData()
        self.__last_input = tensor

        if row_size == 0:
            raise ValueError("RMSNorm cannot work on an empty tensor.")

        # On each row
        for row_start in range(0, len(data), row_size):
            # Get the complete row
            row_end = row_start + row_size
            row = data[row_start:row_end]

            # Calculate RMS
            mean_square = sum(value * value for value in row) / row_size
            rms_x = (mean_square + self.__epsilon) ** 0.5

            # Map each value to value / RMS. Normalization complete
            for x_i in row:
                values.append(x_i / rms_x)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the backward pass for RMS normalization.
        """
        if self.__last_input is None:
            raise ValueError("RMSNorm has not been called yet.")

        values = []
        shape = self.__last_input.getShape()
        input_data = self.__last_input.getData()
        row_size = shape[-1]
        backward_data = backward.getData()

        if row_size == 0:
            raise ValueError("RMSNorm cannot normalize an empty last dimension.")
        if backward.getShape() != shape:
            raise ValueError("backward tensor must have the same shape as the RMSNorm input.")

        # On each row
        for row_start in range(0, len(input_data), row_size):
            # Get the complete rows for input and backward
            row_end = row_start + row_size
            row = input_data[row_start:row_end]
            backward_row = backward_data[row_start:row_end]

            # calculate rms (for 1/rms term)
            mean_square = sum(current_value * current_value for current_value in row) / row_size
            rms = (mean_square + self.__epsilon) ** 0.5

            # gradient for input:
            dot = sum(backward_row[i] * row[i] for i in range(row_size))
            norm_x_sq = sum(row[i] ** 2 for i in range(row_size)) + (row_size * self.__epsilon)

            for i in range(row_size):
                values.append((backward_row[i] - row[i] * dot / norm_x_sq) / rms)

        return Tensor(values, shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the connection is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

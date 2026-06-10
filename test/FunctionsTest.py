import unittest
import math
from Math.Tensor import Tensor
from SequenceProcessing.Functions.AdditionByConstant import AdditionByConstant
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.Mean import Mean
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Functions.RemoveBias import RemoveBias
from SequenceProcessing.Functions.RMSNorm import RMSNorm
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Functions.SiLU import SiLU
from SequenceProcessing.Functions.StableSoftmax import StableSoftmax
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Switch import Switch
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.Variance import Variance


class VarianceTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward computation.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        func = Variance()

        result = func.calculate(tensor)

        # Row-wise mean of squares:
        # row1: (1^2 + 2^2)/2 = (1 + 4)/2 = 2.5
        # row2: (3^2 + 4^2)/2 = (9 + 16)/2 = 12.5
        expected = [2.5, 2.5, 12.5, 12.5]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests derivative.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = Variance()

        result = func.derivative(tensor, backward)

        shape = tensor.getShape()
        expected = []

        for val in tensor.getData():
            expected.append(2.0 * ((shape[1] * val) ** 0.5) / shape[1])

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())


class TransposeTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward transpose operation.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        function = Transpose()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 3.0, 2.0, 4.0], result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests the backward transpose operation.
        """
        value = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))

        function = Transpose()

        result = function.derivative(value, backward)

        self.assertEqual([1.0, 2.0, 3.0, 4.0], result.getData())
        self.assertEqual((2, 2), result.getShape())


class SwitchTest(unittest.TestCase):

    def testCalculateWhenTurnIsTrue(self):
        """
        Tests calculate when turn is True.
        """
        tensor = Tensor([1.0, 2.0, 3.0], (1, 3))
        function = Switch()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 2.0, 3.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testCalculateWhenTurnIsFalse(self):
        """
        Tests calculate when turn is False.
        """
        tensor = Tensor([1.0, 2.0, 3.0], (1, 3))
        function = Switch()
        function.setTurn(False)

        result = function.calculate(tensor)

        self.assertEqual([0.0, 0.0, 0.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivativeWhenTurnIsTrue(self):
        """
        Tests derivative when turn is True.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = Switch()

        result = function.derivative(value, backward)

        self.assertEqual([0.5, 0.6, 0.7], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivativeWhenTurnIsFalse(self):
        """
        Tests derivative when turn is False.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = Switch()
        function.setTurn(False)

        result = function.derivative(value, backward)

        self.assertEqual([0.0, 0.0, 0.0], result.getData())
        self.assertEqual((1, 3), result.getShape())


class SquareRootTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward computation.
        """
        tensor = Tensor([3.0, 8.0, 15.0, 24.0], (2, 2))
        func = SquareRoot(1.0)

        result = func.calculate(tensor)

        # sqrt(1 + x)
        expected = [
            (1 + 3.0) ** 0.5,
            (1 + 8.0) ** 0.5,
            (1 + 15.0) ** 0.5,
            (1 + 24.0) ** 0.5
        ]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = SquareRoot(1.0)

        result = func.derivative(tensor, backward)

        expected = [
            1.0 / (2 * 2.0),
            1.0 / (2 * 4.0),
            1.0 / (2 * 5.0),
            1.0 / (2 * 10.0)
        ]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())



class RemoveBiasTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward computation of RemoveBias.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 99.0], (1, 4))
        function = RemoveBias()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 2.0, 3.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivative(self):
        """
        Tests the backward computation of RemoveBias.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = RemoveBias()

        result = function.derivative(value, backward)

        self.assertEqual([0.5, 0.6, 0.7, 0.0], result.getData())
        self.assertEqual((1, 4), result.getShape())



class TestAdditionByConstant(unittest.TestCase):

    def test_calculate(self):
        tensor = Tensor([1.0, 2.0, 3.0], [3])

        func = AdditionByConstant(2.0)

        result = func.calculate(tensor)

        self.assertEqual(result.getData(), [3.0, 4.0, 5.0])
        self.assertEqual(result.getShape(), [3])

    def test_derivative(self):
        tensor = Tensor([1.0, 2.0, 3.0], [3])
        grad = Tensor([0.5, 0.5, 0.5], [3])

        func = AdditionByConstant(2.0)

        result = func.derivative(tensor, grad)

        self.assertEqual(result.getData(), [0.5, 0.5, 0.5])


class TestMultiplyByConstant(unittest.TestCase):

    def test_calculate(self):
        t = Tensor([1.0, 2.0, 3.0], (1, 3))
        f = MultiplyByConstant(2.0)

        out = f.calculate(t)

        self.assertEqual(out.getData(), [2.0, 4.0, 6.0])
        self.assertEqual(out.getShape(), (1, 3))

    def test_derivative(self):
        t = Tensor([1.0, 2.0, 3.0], (1, 3))
        grad = Tensor([1.0, 1.0, 1.0], (1, 3))

        f = MultiplyByConstant(2.0)

        out = f.derivative(t, grad)

        self.assertEqual(out.getData(), [2.0, 2.0, 2.0])


class InverseTest(unittest.TestCase):

    def testCalculate(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        function = Inverse()

        result = function.calculate(tensor)

        self.assertEqual([0.5, 0.25, 0.2, 0.1], result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        function = Inverse()

        result = function.derivative(tensor, backward)

        self.assertEqual([-4.0, -16.0, -25.0, -100.0], result.getData())
        self.assertEqual((2, 2), result.getShape())




class TestMean(unittest.TestCase):

    def test_calculate(self):
        tensor = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))
        func = Mean()

        result = func.calculate(tensor)

        self.assertEqual(result.getData(), [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(result.getShape(), (2, 2))

    def test_derivative(self):
        tensor = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = Mean()

        result = func.derivative(tensor, backward)

        self.assertEqual(result.getData(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(result.getShape(), (2, 2))



class TestMask(unittest.TestCase):

    def test_calculate(self):
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        func = Mask()

        result = func.calculate(tensor)
        data = result.getData()

        self.assertEqual(data[0], 1.0)
        self.assertTrue(math.isinf(data[1]) and data[1] < 0)
        self.assertEqual(data[2], 3.0)
        self.assertEqual(data[3], 4.0)
        self.assertEqual(result.getShape(), (2, 2))

    def test_derivative(self):
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([0.1, 0.2, 0.3, 0.4], (2, 2))
        func = Mask()

        result = func.derivative(tensor, backward)

        self.assertEqual(result.getData(), [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result.getShape(), (2, 2))


class RotaryPositionEmbeddingTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward RoPE rotation.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4))
        function = RotaryPositionEmbedding()

        result = function.calculate(tensor)
        data = result.getData()

        # Position 0: angle is 0, so the row is unchanged.
        self.assertAlmostEqual(1.0, data[0], 12)
        self.assertAlmostEqual(2.0, data[1], 12)
        self.assertAlmostEqual(3.0, data[2], 12)
        self.assertAlmostEqual(4.0, data[3], 12)

        # Position 1, pair 0: theta = 1 / 10000^(0 / 4) = 1
        theta0 = 1.0
        self.assertAlmostEqual(5.0 * math.cos(theta0) - 6.0 * math.sin(theta0), data[4], 12)
        self.assertAlmostEqual(5.0 * math.sin(theta0) + 6.0 * math.cos(theta0), data[5], 12)

        # Position 1, pair 1: theta = 1 / 10000^(2 / 4)
        theta1 = 1.0 / math.pow(10000.0, 0.5)
        self.assertAlmostEqual(7.0 * math.cos(theta1) - 8.0 * math.sin(theta1), data[6], 12)
        self.assertAlmostEqual(7.0 * math.sin(theta1) + 8.0 * math.cos(theta1), data[7], 12)

        self.assertEqual((2, 4), result.getShape())

    def testDerivative(self):
        """
        Tests that the backward pass applies the inverse rotation, i.e.
        derivative(calculate(x)) recovers the original gradient.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4))
        function = RotaryPositionEmbedding()

        rotated = function.calculate(tensor)
        result = function.derivative(rotated, rotated)

        for expected, actual in zip(tensor.getData(), result.getData()):
            self.assertAlmostEqual(expected, actual, 12)
        self.assertEqual((2, 4), result.getShape())

    def testCalculateRejectsOddLastDimension(self):
        """
        Tests that an odd last dimension is rejected.
        """
        tensor = Tensor([1.0, 2.0, 3.0], (1, 3))
        function = RotaryPositionEmbedding()

        with self.assertRaises(ValueError):
            function.calculate(tensor)


class RMSNormTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward row-wise RMS normalization.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        epsilon = 1e-6
        function = RMSNorm(epsilon)

        result = function.calculate(tensor)
        data = result.getData()

        rms_row1 = ((1.0 + 4.0) / 2 + epsilon) ** 0.5
        rms_row2 = ((9.0 + 16.0) / 2 + epsilon) ** 0.5

        self.assertAlmostEqual(1.0 / rms_row1, data[0], 12)
        self.assertAlmostEqual(2.0 / rms_row1, data[1], 12)
        self.assertAlmostEqual(3.0 / rms_row2, data[2], 12)
        self.assertAlmostEqual(4.0 / rms_row2, data[3], 12)
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests the backward rule of RMS normalization.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        epsilon = 1e-6
        function = RMSNorm(epsilon)

        output = function.calculate(tensor)
        result = function.derivative(output, backward)
        data = result.getData()

        expected = []
        rows = [[1.0, 2.0], [3.0, 4.0]]
        for row in rows:
            n = len(row)
            rms = (sum(v * v for v in row) / n + epsilon) ** 0.5
            dot = sum(1.0 * v for v in row)
            norm_x_sq = sum(v * v for v in row) + n * epsilon
            for v in row:
                expected.append((1.0 - v * dot / norm_x_sq) / rms)

        for e, a in zip(expected, data):
            self.assertAlmostEqual(e, a, 12)
        self.assertEqual((2, 2), result.getShape())

    def testDerivativeWithoutCalculateRaises(self):
        """
        Tests that derivative before calculate raises an error.
        """
        function = RMSNorm()
        backward = Tensor([1.0, 1.0], (1, 2))

        with self.assertRaises(ValueError):
            function.derivative(backward, backward)

    def testInvalidEpsilonRaises(self):
        """
        Tests that a non-positive epsilon is rejected.
        """
        with self.assertRaises(ValueError):
            RMSNorm(0.0)


class SiLUTest(unittest.TestCase):

    @staticmethod
    def sigmoid(value):
        return 1.0 / (1.0 + math.exp(-value))

    def testCalculate(self):
        """
        Tests the forward SiLU activation x * sigmoid(x).
        """
        tensor = Tensor([-1.0, 0.0, 1.0, 2.0], (2, 2))
        function = SiLU()

        result = function.calculate(tensor)
        data = result.getData()

        expected = [v * self.sigmoid(v) for v in [-1.0, 0.0, 1.0, 2.0]]

        for e, a in zip(expected, data):
            self.assertAlmostEqual(e, a, 12)
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests the backward SiLU rule sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
        """
        tensor = Tensor([-1.0, 0.0, 1.0, 2.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        function = SiLU()

        output = function.calculate(tensor)
        result = function.derivative(output, backward)
        data = result.getData()

        expected = []
        for v in [-1.0, 0.0, 1.0, 2.0]:
            s = self.sigmoid(v)
            expected.append(s + v * s * (1.0 - s))

        for e, a in zip(expected, data):
            self.assertAlmostEqual(e, a, 12)
        self.assertEqual((2, 2), result.getShape())

    def testDerivativeWithoutCalculateRaises(self):
        """
        Tests that derivative before calculate raises an error.
        """
        function = SiLU()
        backward = Tensor([1.0, 1.0], (1, 2))

        with self.assertRaises(ValueError):
            function.derivative(backward, backward)


class StableSoftmaxTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward stable softmax over the last dimension.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        function = StableSoftmax()

        result = function.calculate(tensor)
        data = result.getData()

        expected = []
        for row in [[1.0, 2.0], [3.0, 4.0]]:
            maximum = max(row)
            exps = [math.exp(v - maximum) for v in row]
            total = sum(exps)
            expected.extend([e / total for e in exps])

        for e, a in zip(expected, data):
            self.assertAlmostEqual(e, a, 12)
        self.assertEqual((2, 2), result.getShape())

        # Each row must sum to 1.
        self.assertAlmostEqual(1.0, data[0] + data[1], 12)
        self.assertAlmostEqual(1.0, data[2] + data[3], 12)

    def testCalculateIsNumericallyStable(self):
        """
        Tests that large inputs do not overflow.
        """
        tensor = Tensor([1000.0, 1001.0], (1, 2))
        function = StableSoftmax()

        result = function.calculate(tensor)
        data = result.getData()

        expected_sum = 1.0 + math.exp(1.0)
        self.assertAlmostEqual(1.0 / expected_sum, data[0], 12)
        self.assertAlmostEqual(math.exp(1.0) / expected_sum, data[1], 12)

    def testDerivative(self):
        """
        Tests the backward softmax rule y * (g - sum_j y_j * g_j).
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([0.1, 0.2, 0.3, 0.4], (2, 2))
        function = StableSoftmax()

        output = function.calculate(tensor)
        result = function.derivative(output, backward)
        data = result.getData()

        output_data = output.getData()
        backward_data = backward.getData()

        expected = []
        for start in (0, 2):
            y = output_data[start:start + 2]
            g = backward_data[start:start + 2]
            total = sum(yi * gi for yi, gi in zip(y, g))
            expected.extend([yi * (gi - total) for yi, gi in zip(y, g)])

        for e, a in zip(expected, data):
            self.assertAlmostEqual(e, a, 12)
        self.assertEqual((2, 2), result.getShape())

        # Gradient entries of a softmax row sum to zero when projected through y.
        self.assertAlmostEqual(0.0, data[0] + data[1], 6)


if __name__ == "__main__":
    unittest.main()

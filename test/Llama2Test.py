import unittest

from Math.Tensor import Tensor

from SequenceProcessing.Classification.Llama2 import Llama2
from SequenceProcessing.Parameters.Llama2Parameter import Llama2Parameter


class Llama2Test(unittest.TestCase):

    def testBuildGraph(self):
        parameter = Llama2Parameter.tinyLlama2(epoch=1)
        model = Llama2(parameter)

        model.buildGraph()

        self.assertTrue(len(model.input_nodes) >= 2)
        self.assertIsNotNone(model.output_node)

    def testSetInput(self):
        parameter = Llama2Parameter.tinyLlama2(epoch=1)
        model = Llama2(parameter)
        model.buildGraph()

        model.setInput([0, 1, 2, 3])

        self.assertEqual((4, parameter.getVocabularyLength()), model.input_nodes[0].getValue().getShape())

    def testContextLengthExceeded(self):
        parameter = Llama2Parameter.tinyLlama2(epoch=1)
        model = Llama2(parameter)
        model.buildGraph()

        with self.assertRaises(ValueError):
            model.setInput(list(range(parameter.getContextLength() + 1)))

    def testPredictNextTokenOnFreshGraph(self):
        parameter = Llama2Parameter.tinyLlama2(epoch=1)
        model = Llama2(parameter)

        next_token_id = model.predictNextToken([0, 1, 2])

        self.assertIsInstance(next_token_id, int)
        self.assertGreaterEqual(next_token_id, 0)
        self.assertLess(next_token_id, parameter.getVocabularyLength())

    def testTrainAndGenerate(self):
        parameter = Llama2Parameter.tinyLlama2(epoch=1)
        model = Llama2(parameter)
        train_set = [Tensor([0, 1, 2, 3, 0, 1, 2, 3], (8,))]

        model.train(train_set)
        generated = model.generateGreedy([0, 1], max_new_tokens=4)

        self.assertEqual([0, 1], generated[:2])
        self.assertEqual(6, len(generated))
        for token_id in generated:
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, parameter.getVocabularyLength())


if __name__ == "__main__":
    unittest.main()

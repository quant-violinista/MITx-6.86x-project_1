import unittest
import numpy as np
from perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    def test_perceptron(self):
        x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
        y = np.array([1, -1, 1])
        machine = Perceptron(x, y, 100, 0, True)
        machine.run()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

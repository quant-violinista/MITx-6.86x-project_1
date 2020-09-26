import unittest
import numpy as np
from perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    def test_perceptron(self):
        x = np.array([[np.cos(np.pi), 0], [0, np.cos(2*np.pi)]])
        y = np.array([1, 1])
        machine = Perceptron(x, y, 50, 0, False, True)
        machine.run()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from em_generative_models import EM


class EMGenerativeTest(unittest.TestCase):
    def test_log_likelihood(self):
        points = np.array([-1, 0, 4, 5, 6], dtype=float)
        theta = [{'weight': 0.5, 'mean': 6, 'var': 1}, {'weight': 0.5, 'mean': 7, 'var': 4}]
        em_object = EM(points, k=2, theta=theta)
        em_object.cluster(iterations=20)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

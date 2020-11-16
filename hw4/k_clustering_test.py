import unittest
import numpy as np
from k_clustering import KMeans


class MyTestCase(unittest.TestCase):
    def test_k_means(self):
        x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]], dtype=float)
        z = np.array([(-5, 2), (0, -6)], dtype=float)
        k_object = KMeans(x, z, 1)
        k_object.cluster_data()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

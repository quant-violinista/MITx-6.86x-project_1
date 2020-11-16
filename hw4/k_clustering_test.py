import unittest
import numpy as np
from k_clustering import KMeans, KMediods


class ClusteringTest(unittest.TestCase):
    def test_k_means_explicit(self):
        x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]], dtype=float)
        z = np.array([(-5, 2), (0, -6)], dtype=float)
        k_object = KMeans(x, z=z, norm=1)
        k_object.cluster_data()
        self.assertEqual(True, True)

    def test_k_means_random(self):
        x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]], dtype=float)
        k_object = KMeans(x, k=2, norm=1)
        k_object.cluster_data()
        self.assertEqual(True, True)

    def test_k_mediods_explicit(self):
        x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]], dtype=float)
        z = np.array([(-5, 2), (0, -6)], dtype=float)
        k_object = KMediods(x, z=z, norm=1)
        k_object.cluster_data()
        self.assertEqual(True, True)

    def test_k_mediods_explicit_euclidian(self):
        x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]], dtype=float)
        z = np.array([(-5, 2), (0, -6)], dtype=float)
        k_object = KMediods(x, z=z, norm=2)
        k_object.cluster_data()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()

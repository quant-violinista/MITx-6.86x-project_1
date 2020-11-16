import numpy as np
import pandas as pd


class K:
    def __init__(self, x, z, norm):
        self.x = x
        self.z = z
        self.norm = norm
        self.k = len(z)
        self.cost = None
        self.clusters = None

    def assign_cluster(self):
        indices = []
        cost = 0
        for point in self.x:
            cost += np.min(np.sum(np.abs(self.z - point) ** self.norm, axis=1))
            indices.append(np.argmin(np.sum(np.abs(self.z - point) ** self.norm, axis=1)))
        self.cost = cost
        self.clusters = pd.DataFrame(self.x, index=indices)


class KMeans(K):
    def __init__(self, x, z, norm=2):
        super().__init__(x, z, norm)

    def update_representatives(self):
        clusters = self.clusters.groupby(axis=0, level=0)
        if self.norm == 1:
            self.z = clusters.aggregate(np.median).to_numpy()
        else:
            self.z = clusters.aggregate(np.average).to_numpy()

    def cluster_data(self, iterations=1000, debug=True):
        assert self.norm in [1, 2], 'Only L1 and L2 norms allowed.'
        for i in range(iterations):
            self.assign_cluster()
            self.update_representatives()
        if debug:
            print(f'The cost : {self.cost}')
            print(f'The representatives are : {self.z}')
            print(self.clusters)

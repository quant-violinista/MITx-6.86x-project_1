import numpy as np
import pandas as pd
import random


class K:
    def __init__(self, x, z, k, norm=2):
        assert (z is not None) | (k is not None), 'Specify either k or initial representatives explicitly'
        self.x = x
        self.z = z if z is not None else random.choices(x, k=k)
        self.norm = norm
        self.k = len(self.z)
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
    def __init__(self, x, z=None, k=None, norm=2):
        super().__init__(x, z, k, norm)

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


class KMediods(K):
    def __init__(self, x, z=None, k=None, norm=2):
        super().__init__(x, z, k, norm)

    def update_representatives(self):
        clusters = self.clusters.groupby(axis=0, level=0)
        for i, group in clusters:
            representative = self.z[i]
            cost = 10000000.
            for _, row in group.iterrows():
                cost_row = np.sum(np.abs(row - group) ** self.norm, axis=1)
                cost_row = np.sum(cost_row, axis=0)
                if cost_row < cost:
                    cost = cost_row
                    representative = row
            self.z[i] = representative

    def cluster_data(self, iterations=1000, debug=True):
        for i in range(iterations):
            self.assign_cluster()
            self.update_representatives()
        if debug:
            print(f'The cost : {self.cost}')
            print(f'The representatives are : {self.z}')
            print(self.clusters)

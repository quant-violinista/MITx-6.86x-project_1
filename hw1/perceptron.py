import numpy as np


class Perceptron:
    def __init__(self, x, y, iterations, start_idx=0, debug=False):
        """
        :param x: numpy array containing feature vectors
        :param y: numpy array containing labels
        :param iterations: number of iterations through points
        :param start_idx: index of point to start iteration
        :param debug: option to print output
        """
        self.num_points = x.shape[0]
        assert self.num_points == y.size
        assert iterations >= 1
        self.x = x
        self.y = y
        self.iterations = iterations
        self.theta = np.zeros(x.shape[1])
        self.theta0 = 0.
        self.debug = debug
        self.start_idx = start_idx

    def __no_mistakes__(self):
        if np.all(np.sign(self.x.dot(self.theta) + self.theta0) == self.y):
            return True
        else:
            return False

    def run(self):
        for i in range(self.iterations):
            for j in range(self.num_points):
                if (i == 0) & (j < self.start_idx):
                    continue
                if np.sign(self.x[j].dot(self.theta) + self.theta0) != self.y[j]:
                    self.theta += self.y[j] * self.x[j]
                    self.theta0 += self.y[j]
                    if self.debug:
                        print(self.theta)
                        print(self.theta0)
                if self.__no_mistakes__():
                    return

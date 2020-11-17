import numpy as np
from scipy.stats import norm
import pandas as pd


class EM:
    def __init__(self, x, k, theta):
        self.x = x
        self.k = k
        self.theta = pd.DataFrame(theta)
        self.posteriors = None

    def compute_log_likelihood(self):
        probabilities = np.zeros(np.shape(self.x))
        for _, theta in self.theta.iterrows():
            probabilities += theta['weight'] * norm.pdf(self.x, theta['mean'], np.sqrt(theta['var']))

        log_likelihood = np.sum(np.log(probabilities))

        return log_likelihood

    def compute_posteriors(self):
        probabilities = []
        for _, theta in self.theta.iterrows():
            probabilities.append(theta['weight'] * norm.pdf(self.x, theta['mean'], np.sqrt(theta['var'])))

        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities, axis=0)
        self.posteriors = probabilities
        return

    def maximization(self):
        weight = np.sum(self.posteriors, axis=1) / len(self.x)
        mean = np.sum(self.posteriors * self.x, axis=1) / np.sum(self.posteriors, axis=1)
        var = np.sum((np.transpose(np.tile(self.x, (mean.shape[0], 1)).T - mean) ** 2) * self.posteriors,
                     axis=1) / np.sum(self.posteriors, axis=1)

        self.theta = pd.DataFrame({'weight': weight, 'mean': mean, 'var': var})
        return

    def cluster(self, iterations=100):
        for i in range(iterations):
            self.compute_posteriors()
            self.maximization()
            print(f'*****************  iteration: {i}  *******************')
            print(self.theta)

        return

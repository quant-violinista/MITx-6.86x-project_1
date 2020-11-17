import numpy as np
from scipy.stats import norm


class EM:
    def __init__(self, x, k, theta):
        self.x = x
        self.k = k
        self.theta = theta

    def compute_log_likelihood(self):
        probabilities = np.zeros(np.shape(self.x))
        for gaussian in self.theta:
            probabilities += gaussian['weight'] * norm.pdf(self.x, gaussian['mean'], np.sqrt(gaussian['var']))

        log_likelihood = np.sum(np.log(probabilities))

        return log_likelihood

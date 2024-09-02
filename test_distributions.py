import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, vonmises
import utils

import unittest

class TestDistributions(unittest.TestCase):
    def test_gaussian_samples(self):
        mus = np.array([-3, -1.5, 0, 1, 2])  
        sigmas = np.array([1, 2, 3])  
        n, m = len(mus), len(sigmas)
        num_samples = 100000

        # Generate samples using broadcasting
        samples = np.random.normal(loc=mus[:, np.newaxis], scale=sigmas[np.newaxis, :],
                                    size=(num_samples, n, m))

        # Calculate empirical means and standard deviations for each combination
        empirical_means = np.mean(samples, axis=0)
        empirical_stddevs = np.std(samples, axis=0)
        for i, mu in enumerate(mus):
            for j, sigma in enumerate(sigmas):
                self.assertAlmostEqual(empirical_means[i,j], mu, places=1)
                self.assertAlmostEqual(empirical_stddevs[i,j], sigma, places=1)
    
    def test_von_mises_samples(self):
        mus = np.array([-3, -1, 0, 1, 2])  # Example means (length n)
        kappas = np.array([.5, 1, 2, 3, 50, 100])  # Example standard deviations (length m)
        num_samples = 10000  # Number of samples to generate for each distribution

        # Generate samples using broadcasting with 2D arrays
        # Here, mus[:, np.newaxis] reshapes mus to (n, 1) and sigmas[np.newaxis, :] reshapes sigmas to (1, m)
        samples = vonmises.rvs(loc=mus[:, np.newaxis], kappa=kappas[np.newaxis, :], 
                                size=(num_samples, len(mus), len(kappas)))
        est_mus, est_kappas = utils.estimate_mu_and_kappa_von_mises(samples, axis=0)
        for i, mu in enumerate(mus):
            for j, kappa in enumerate(kappas):
                print(f'True mu, kappa={mu, kappa} vs estimated={est_mus[i,j], est_kappas[i,j]}')


if __name__ == '__main__':
    unittest.main()
        

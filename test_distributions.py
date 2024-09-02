import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, vonmises
import utils

import unittest

def plot_samples(samples, mus, sigmas, dist):
    fig, axes = plt.subplots(len(mus), len(sigmas), figsize=(15, 10))

    for i in range(len(mus)):
        for j in range(len(sigmas)):
            ax = axes[i, j]

            # Plot histogram of the samples
            ax.hist(samples[:, i, j], bins=50, density=True, alpha=0.6, color='g')
            
            # Plot the theoretical normal distribution curve
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            if dist == 'VM':
                p = vonmises.pdf(x,loc= mus[i], kappa=sigmas[j])
            else: 
                p = norm.pdf(x, loc=mus[i], scale=sigmas[j])
            ax.plot(x, p, 'k', linewidth=2)
            
            title = f"VM mu={mus[i]}, kappa={sigmas[j]}"
            ax.set_title(title)

    plt.tight_layout()
    plt.show()

class TestDistributions(unittest.TestCase):
    def test_gaussian_samples(self):
        mus = np.array([-3, -1.5, 0, 1, 2])  
        sigmas = np.array([1, 2, 3])  
        n, m = len(mus), len(sigmas)
        num_samples = 100000

        # Generate samples using broadcasting
        samples = np.random.normal(loc=mus[:, np.newaxis], scale=sigmas[np.newaxis, :],
                                    size=(num_samples, n, m))
        plot_samples(samples, mus, sigmas, dist='Gaussian')
        # Calculate empirical means and standard deviations for each combination
        empirical_means = np.mean(samples, axis=0)
        empirical_stddevs = np.std(samples, axis=0)
        for i, mu in enumerate(mus):
            for j, sigma in enumerate(sigmas):
                self.assertAlmostEqual(empirical_means[i,j], mu, places=1)
                self.assertAlmostEqual(empirical_stddevs[i,j], sigma, places=1)
    
    def test_von_mises_samples(self):
        mus = np.array([-3, -1, 0, 1, 2])  # Example means (length n)
        kappas = np.array([.5, 1, 2, 3, 50, 100])  # Example concentrations (length m)
        num_samples = 10000  # Number of samples to generate for each distribution

        # Generate samples using broadcasting with 2D arrays
        # mus[:, np.newaxis] reshapes mus to (n, 1) and kappas[np.newaxis, :] reshapes kappas to (1, m)
        samples = vonmises.rvs(loc=mus[:, np.newaxis], kappa=kappas[np.newaxis, :], 
                                size=(num_samples, len(mus), len(kappas)))
        est_mus, est_kappas = utils.estimate_mu_and_kappa_von_mises(samples, axis=0)
        for i, mu in enumerate(mus):
            for j, kappa in enumerate(kappas):
                print(f'True mu, kappa={mu, kappa} vs estimated={est_mus[i,j], est_kappas[i,j]}')

    def test_von_mises_samples_kappa_matrix(self):
        mus = np.array([-3, -1, 0, 1, 2])  # Example means (length n)
        kappas = np.array([[.5, 1, 2, 3, 50, 100],
                           [1.2, 2, 2, 3, 20, 50],
                           [.3, 10, 2, 3, 5, 80],
                           [.5, 1, 2, 4, 60, 40],
                           [1, .3, 2, 3, 50, 100]])  # Example concentrations (n, m)
        num_samples = 10000  # Number of samples to generate for each distribution
        samples = vonmises.rvs(loc=mus[:, np.newaxis], kappa=kappas, 
                                size=(num_samples, len(mus), kappas.shape[1]))
        est_mus, est_kappas = utils.estimate_mu_and_kappa_von_mises(samples, axis=0)
        for i, mu in enumerate(mus):
            for j, kappa in enumerate(kappas[i]):
                print(f'True mu, kappa={mu, kappa} vs estimated={est_mus[i,j], est_kappas[i,j]}')


if __name__ == '__main__':
    unittest.main()
        

import numpy as np
import causal_inference
import utils
from scipy.stats import vonmises
from scipy.special import i0
import matplotlib.pyplot as plt

def get_cue_combined_mean_params(mu1, kappa1, mu2, kappa2):
    """
    Determine the mean parameters of the product densities of two von Mises distributions.

    Parameters:
    mu1 (float): Mean direction of the first von Mises distribution.
    kappa1 (float): Concentration parameter of the first von Mises distribution.
    mu2 (float): Mean direction of the second von Mises distribution.
    kappa2 (float): Concentration parameter of the second von Mises distribution.

    Returns:
    tuple: Combined mean direction and concentration parameter.
    """
    mu = mu2 + np.arctan2(np.sin(mu1-mu2), kappa2/kappa1 + np.cos(mu1-mu2))
    k = np.sqrt((kappa1**2) + (kappa2**2) + 2*kappa1*kappa2*np.cos(mu1 - mu2))
    return mu, k

def product_of_von_Mises(mu1, kappa1, mu2, kappa2, plot=False):
    """
    Compute the product of two von Mises densities and compare with the analytic combined distribution.

    Parameters:
    mu1 (float): Mean direction of the first von Mises distribution.
    kappa1 (float): Concentration parameter of the first von Mises distribution.
    mu2 (float): Mean direction of the second von Mises distribution.
    kappa2 (float): Concentration parameter of the second von Mises distribution.
    plot (bool): Whether to plot the normalized product and combined density. Default is False.

    Returns:
    None
    """
    mu_c, kappa_c = get_cue_combined_mean_params(mu1, kappa1, mu2, kappa2)
    x_domain = np.linspace(-np.pi, np.pi, 100000)
    product_pdf = vonmises.pdf(x=x_domain, loc=mu1, kappa=kappa1) * vonmises.pdf(x=x_domain, loc=mu2, kappa=kappa2)
    combined_pdf = vonmises.pdf(x=x_domain, loc=mu_c, kappa=kappa_c)
    assert np.allclose(product_pdf / np.sum(product_pdf), combined_pdf / np.sum(combined_pdf), atol=1e-5)
    if plot:
        plt.plot(x_domain, product_pdf / np.sum(product_pdf), label='normalised product')
        plt.plot(x_domain, combined_pdf / np.sum(combined_pdf), linestyle='--', label='combined density')
        plt.title(f'Product of densities VM({mu1, kappa1}), VM({mu2, kappa2})')
        plt.legend()
        plt.show()

class VonMisesCausalInference(causal_inference.CausalInference):
    def __init__(self, simulate=False):
        super().__init__(distribution='vonMises')
        self.simulate = simulate
        self.s_domain = np.linspace(-np.pi, np.pi, 1000).reshape(1, 1, 1, -1)

    def fusion_estimate(self, x_a, x_v, sigma_a, sigma_v, mu_p, sigma_p, simulate=False, return_sigma=False):
        """
        Compute the MAP estimate in the fusion case (combined sensory processing).
        The posterior p(s|x_a, x_v) \propto p(x_a, x_v|s)p(s) = p(x_a|s)p(x_v|s)p(s) with p(s) as the
        prior and p(x_a|s), p(x_v|s) as the likelihoods.
        The prior is currently assumed uniform and its parameters are not used.

        Parameters:
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        sigma_a (float): Auditory sensory noise (concentration).
        sigma_v (float): Visual sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Fusion estimate.
        """
        assert (simulate == False)
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_a, kappa1=sigma_a, mu2=x_v, kappa2=sigma_v)
        if return_sigma:
            return mu_c, kappa_c
        return mu_c
    
    def fusion_posterior_params(self, s_a, s_v, sigma_a, sigma_v, mu_p, sigma_p):
        fused_est_mu, fused_est_sigma  = self.fusion_estimate(x_a=s_a, x_v=s_v, 
                                                            sigma_a=sigma_a, 
                                                            sigma_v=sigma_v, 
                                                            mu_p=mu_p, 
                                                            sigma_p=sigma_p,
                                                            return_sigma=True)
        return fused_est_mu, fused_est_sigma
    
    def segregation_estimate(self, x, mu_p, sigma, sigma_p):
        pass

    def sim_likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        """
        Compute the likelihood of the common cause hypothesis (signals from the same source).
        p(x_v, x_a|C=1) = \int p(x_v, x_a|s)p(s)ds = \int p(x_v|s)p(x_a|s)p(s)ds
        The function uses numeric integration.
        The prior is currently assumed uniform and its parameters are not used.

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Likelihood of the common cause hypothesis.
        """
        print('Computing p(x_V, x_A| C=1) using numerical integration and sampled x_V, x_A')
        p_x_v_given_s = vonmises.pdf(x=x_a[..., np.newaxis], loc=self.s_domain, kappa=sigma_a)
        p_x_a_given_s = vonmises.pdf(x=x_v[..., np.newaxis], loc=self.s_domain, kappa=sigma_v)
        # p_s = 1 / len(self.s_domain)
        p_s = 1 # using uniform prior
        return np.trapz(p_x_v_given_s*p_x_a_given_s*p_s, axis=-1, x=self.s_domain)

    def likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        """
        Compute the likelihood of the common cause hypothesis (signals from the same source).
        p(x_v, x_a|C=1) = \int p(x_v, x_a|s)p(s)ds = \int p(x_v|s)p(x_a|s)p(s)ds
        The prior is currently assumed uniform and its parameters are not used.

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Likelihood of the common cause hypothesis.
        """
        if self.simulate:
            return self.sim_likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        print('Computing p(x_V, x_A| C=1) using analytic solution on sampled x_V, x_A')
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_a, mu2=x_v, kappa1=sigma_a, kappa2=sigma_v)
        return i0(kappa_c) / (2*np.pi*i0(sigma_a)*i0(sigma_v))

    def likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        pass

    def posterior_prob_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass

    def bayesian_causal_inference(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass

    
    def generate_samples_common_cause(self, n_means, n_samples, kappa_1, kappa_2):
        """
        Generate samples x_v, x_a assuming a common cause for the two sensory inputs.

        Parameters:
        n_means (int): Number of mean directions to generate.
        n_samples (int): Number of samples per mean direction.
        kappa_1 (np.ndarray): Concentration parameters for the first stimulus von Mises distributions.
        kappa_2 (np.ndarray): Concentration parameters for the second stimulus von Mises distributions.

        Returns:
        tuple: Generated mean directions and samples for the two sensory inputs.
        - s (np.ndarray): True stimulus values of size (n_means) with uniform values in [-pi, pi].
        - x_vs (np.ndarray): Samples for the first sensory input of shape (n_means, n_samples).
        - x_as (np.ndarray): Samples for the second sensory input of shape (n_means, n_samples).
        """
        # Generate s_v and s_a of size n_means with uniform values in [-pi, pi]
        s = np.random.uniform(low=-np.pi, high=np.pi, size=n_means)

        # Generate x_vs and x_as with shape (n_means, n_samples) of Von Mises samples
        x_vs = np.zeros((n_means, n_samples))
        x_as = np.zeros((n_means, n_samples))

        for i in range(n_means):
            x_vs[i] = utils.wrap(vonmises(loc=s[i], kappa=kappa_1[i]).rvs(size=n_samples))
            x_as[i] = utils.wrap(vonmises(loc=s[i], kappa=kappa_2[i]).rvs(size=n_samples))

        return s, x_vs, x_as
    
    def generate_samples_two_causes(self, n_means, n_samples, kappa_1, kappa_2):
        """
        Generate samples x_v, x_a assuming two independent causes for the two sensory inputs.

        Parameters:
        n_means (int): Number of mean directions to generate.
        n_samples (int): Number of samples per mean direction.
        kappa_1 (np.ndarray): Concentration parameters for the first stimulus von Mises distributions.
        kappa_2 (np.ndarray): Concentration parameters for the second stimulus von Mises distributions.

        Returns:
        tuple: Generated mean directions and samples for the two sensory inputs.
        - s_v (np.ndarray): True stimulus values for the first cause of size (n_means) with uniform values in [-pi, pi].
        - s_a (np.ndarray): True stimulus values for the second cause of size (n_means) with uniform values in [-pi, pi].
        - x_vs (np.ndarray): Samples for the first sensory input of shape (n_means, n_samples).
        - x_as (np.ndarray): Samples for the second sensory input of shape (n_means, n_samples).
        """
        # Generate s_v and s_a of size n_means with uniform values in [-pi, pi]
        s_v = np.random.uniform(low=-np.pi, high=np.pi, size=n_means)
        s_a = np.random.uniform(low=-np.pi, high=np.pi, size=n_means)

        # Generate x_vs and x_as with shape (n_means, n_samples) of Von Mises samples
        x_vs = np.zeros((n_means, n_samples))
        x_as = np.zeros((n_means, n_samples))

        for i in range(n_means):
            x_vs[i] = utils.wrap(vonmises(loc=s_v[i], kappa=kappa_1[i]).rvs(size=n_samples))
            x_as[i] = utils.wrap(vonmises(loc=s_a[i], kappa=kappa_2[i]).rvs(size=n_samples))

        return s_v, s_a, x_vs, x_as

    def generate_samples_causal_inference(self, p_common, n_means, n_samples, kappa_1, kappa_2):
        ps = np.random.binomial(n=1, size=n_samples, p=p_common)
        n_common_cause = sum(ps)
        _, _, x_vs_C2, x_as_C2 = self.generate_samples_two_causes(n_means, n_samples-n_common_cause, kappa_1, kappa_2)
        _, x_vs_C1, x_as_C1 = self.generate_samples_common_cause(n_means, n_common_cause, kappa_1, kappa_2)
        x_vs = np.concatenate([x_vs_C1.flatten(), x_vs_C2.flatten()])
        x_as = np.concatenate([x_as_C1.flatten(), x_as_C2.flatten()])
        return x_vs, x_as

num_sim = 1000
stimuli_values = np.linspace(-np.pi, np.pi, 5)  # Von Mises distribution is defined on the interval [-pi, pi]
s_vs, s_as = np.meshgrid(stimuli_values, stimuli_values, indexing='ij')

# Parameters for the von Mises distribution
kappa_v, kappa_a = 1 / (2.14), 1 / (9.2)  # Concentration parameters for visual and auditory inputs
product_of_von_Mises(mu1=stimuli_values[0], mu2=stimuli_values[2], kappa1=kappa_v, kappa2=kappa_a)
mu_p, kappa_p = 0, 1 / (12.3)  # Prior mean direction and concentration parameter
pi_c = 0.23  # Prior probability of the common cause hypothesis
print(f'Svs = {s_vs.reshape(-1)}\nSas = {s_as.reshape(-1)}\n')

# Generate random samples for each combination of cues
x_v = vonmises.rvs(kappa=kappa_v, loc=s_vs, size=(num_sim, stimuli_values.size, stimuli_values.size))
x_a = vonmises.rvs(kappa=kappa_a, loc=s_as, size=(num_sim, stimuli_values.size, stimuli_values.size))


model = VonMisesCausalInference()
# Generate samples (as described in "Generative model" in Kording, 2007) assuming the probability 
# of a common cause follows a Bernoulli distribution.
n_means = 100
n_samples = 1000000
x_vs, x_as = model.generate_samples_causal_inference(p_common=.28, n_means=n_means, n_samples=n_samples,
                                                    kappa_1=np.ones(n_means),
                                                    kappa_2=np.ones(n_means))
utils.plot_2d_histogram(x_vs, x_as)

# Compute the posterior estimates using simulation
fused_est = model.fusion_estimate(x_v, x_a, kappa_a, kappa_v, mu_p, kappa_p)
# Compute the posterior estimates using the VM approx
fused_est_mu, fused_est_kappa = model.fusion_posterior_params(s_a=s_as, s_v=s_vs, 
                                                              sigma_a=kappa_a, sigma_v=kappa_v, 
                                                              mu_p=mu_p, sigma_p=kappa_p)

# Generate analytic samples
fused_est_approx = vonmises.rvs(kappa=fused_est_kappa, loc=fused_est_mu,
                                  size=(num_sim, stimuli_values.size, stimuli_values.size))

# Plot histograms for comparison
plt.hist(fused_est_approx[:, 1, 1], bins=20, label='approx', alpha=0.5, edgecolor='b', histtype='step', density=True)
plt.hist(fused_est[:, 1, 1], bins=20, label='numeric', alpha=0.5, edgecolor='r', histtype='step', density=True)
plt.legend()
plt.title('Von Mises approximation and simulated distribution of mean responses')
plt.show()

sim_model = VonMisesCausalInference(simulate=True)
# compute p(x_V, x_A| C=1) by simulating x_V, x_A and using equation (4) in Kording, 2007
likelihood_common_cause = model.likelihood_common_cause(x_v=x_v, x_a=x_a, sigma_v=kappa_v, 
                                                      sigma_a=kappa_a, mu_p=mu_p, sigma_p=kappa_p)
# compute p(x_V, x_A| C=1) = \int p(x_V|s) p(x_A|s) p(s) ds by simulating x_V, x_A and numerical integration
sim_likelihood_common_cause = sim_model.likelihood_common_cause(x_v=x_v, x_a=x_a, 
                                                                sigma_v=kappa_v, sigma_a=kappa_a,
                                                                mu_p=mu_p, sigma_p=kappa_p)

diff_likelihood_common_cause=likelihood_common_cause-sim_likelihood_common_cause
print(f'Max difference between analytic and simulated likelihood: {np.max(np.abs(diff_likelihood_common_cause))}')
print(f'Max analytic and simulated likelihood: {np.max(likelihood_common_cause), np.max(sim_likelihood_common_cause)}')
plt.hist(likelihood_common_cause[:, 1,1], bins=20, label='analytic', alpha=0.5, edgecolor='b', histtype='step', density=True)
plt.hist(sim_likelihood_common_cause[:, 1, 1], bins=20, label='sim', alpha=0.5, edgecolor='r', histtype='step', density=True)
plt.legend()
plt.show()

import numpy as np
import causal_inference
from scipy.stats import vonmises
from scipy.special import i0
import matplotlib.pyplot as plt

def get_cue_combined_mean_params(mu1, kappa1, mu2, kappa2):
    mu = mu2 + np.arctan2(np.sin(mu1-mu2), kappa2/kappa1 + np.cos(mu1-mu2))
    k = np.sqrt((kappa1**2) + (kappa2**2) + 2*kappa1*kappa2*np.cos(mu1 - mu2))
    return mu, k

def product_of_von_Mises(mu1, kappa1, mu2, kappa2, plot=False):
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
        # res = \int_{s in s_domain} N(x_a; s, sigma_a) * N(x_a; s, sigma_a) * p(s)
        p_x_v_given_s = vonmises.pdf(x=x_a[..., np.newaxis], loc=self.s_domain, kappa=sigma_a)
        p_x_a_given_s = vonmises.pdf(x=x_v[..., np.newaxis], loc=self.s_domain, kappa=sigma_v)
        # p_s = 1 / len(self.s_domain)
        p_s = 1 # using uniform prior
        return np.trapz(p_x_v_given_s*p_x_a_given_s*p_s, axis=-1, x=self.s_domain)

    def likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        """
        Compute the likelihood of the common cause hypothesis (signals from the same source).
        p(x_v, x_a|C=1) = \int p(x_v, x_a|s)p(s)ds = \int p(x_v|s)p(x_a|s)p(s)ds

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (kappa).
        sigma_a (float): Auditory sensory noise (kappa).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior standard deviation of the stimulus rate.

        Returns:
        (float or np.ndarray): Likelihood of the common cause hypothesis.
        """
        if self.simulate:
            return self.sim_likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_a, mu2=x_v, kappa1=sigma_a, kappa2=sigma_v)
        return i0(kappa_c) / (2*np.pi*i0(sigma_a)*i0(sigma_v))

    def likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        pass

    def posterior_prob_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass

    def bayesian_causal_inference(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass


num_sim = 10000
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
plt.show()

sim_model = VonMisesCausalInference(simulate=True)
# compute p(x_V, x_A| C=1) by simulating x_V, x_A and using equation (4) in Kording, 2007
likelihood_common_cause = model.likelihood_common_cause(x_v=x_v, x_a=x_a, sigma_v=kappa_v, 
                                                      sigma_a=kappa_a, mu_p=mu_p, sigma_p=kappa_p)
# compute p(x_V, x_A| C=1) = \int p(x_V|s) p(x_A|s) p(s) ds by simulating x_V, x_A and numerical int
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
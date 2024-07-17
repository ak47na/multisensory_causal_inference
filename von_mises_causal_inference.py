import numpy as np
import causal_inference
import utils
from scipy.stats import vonmises, circmean
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
    mu = utils.wrap(mu2 + np.arctan2(np.sin(mu1-mu2), kappa2/kappa1 + np.cos(mu1-mu2)))
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

    def fusion_estimate(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, simulate=False, return_sigma=False):
        """
        Compute the MAP estimate in the fusion case (combined sensory processing).
        The posterior p(s|x_v, x_a) \propto p(x_v, x_a|s)p(s) = p(x_a|s)p(x_v|s)p(s) with p(s) as the
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
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises fusion estimate only implemented for uniform prior")
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_a, kappa1=sigma_a, mu2=x_v, kappa2=sigma_v)
        if return_sigma:
            return mu_c, kappa_c
        return mu_c
    
    def fusion_posterior_params(self, s_v, s_a, sigma_v, sigma_a, mu_p, sigma_p):
        fused_est_mu, fused_est_sigma  = self.fusion_estimate(x_v=s_v, x_a=s_a, 
                                                            sigma_a=sigma_a, 
                                                            sigma_v=sigma_v, 
                                                            mu_p=mu_p, 
                                                            sigma_p=sigma_p,
                                                            return_sigma=True)
        return fused_est_mu, fused_est_sigma
    
    def segregation_estimate(self, x, mu_p, sigma, sigma_p):
        """
        Compute the MAP estimate in the segregation case (independent sensory processing).
        The posterior p(s|x) \propto p(x|s)p(s), with p(s) as the prior and p(x|s) as the likelihood.
        For uniform p(s), because the Von Mises is symmetric, the MAP is x.

        Parameters:
        x (float or np.ndarray): Sensory input (e.g., observed rate).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma (float): Sensory noise (concentration).
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Segregation estimate matching the type of x.
        """
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises segregation estimate only implemented for uniform prior")
        return x

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
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises common cause likelihood only implemented for uniform prior")
        p_x_v_given_s = vonmises.pdf(x=x_a[..., np.newaxis], loc=self.s_domain, kappa=sigma_a)
        p_x_a_given_s = vonmises.pdf(x=x_v[..., np.newaxis], loc=self.s_domain, kappa=sigma_v)
        # p_s = 1 / len(self.s_domain)
        p_s = 1 / (2*np.pi) # using uniform prior
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
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises common cause likelihood only implemented for uniform prior")
        print('Computing p(x_V, x_A| C=1) using analytic solution on sampled x_V, x_A')
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_a, mu2=x_v, kappa1=sigma_a, kappa2=sigma_v)
        return i0(kappa_c) / (((2*np.pi)**2)*i0(sigma_a)*i0(sigma_v))
    
    def sim_likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        """
        Compute the likelihood of the separate causes hypothesis (signals from different sources).
        p(x_v, x_a|C=2) = \int \int p(x_v, x_a|s_v, s_a)p(s_v)p(s_a)ds_v ds_a and due to independence
                        = (\int p(x_v|s_v)p(s_v) ds_v)(\int p(x_a|s_a)p(s_a) ds_a)
        Currently, only uniform priors are supported, hence p(x_v, x_a|C=2) = \frac{1}{2\pi}^2
        The function uses numeric integration.

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Likelihood of the separate causes hypothesis.
        """
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises separate cause likelihood only implemented for uniform prior")
        print('Computing p(x_V, x_A| C=2) using numerical integration and sampled x_V, x_A')
        p_x_v_given_s = vonmises.pdf(x=x_a[..., np.newaxis], loc=self.s_domain, kappa=sigma_a)
        p_x_a_given_s = vonmises.pdf(x=x_v[..., np.newaxis], loc=self.s_domain, kappa=sigma_v)
        # p_s = 1 / len(self.s_domain)
        p_s = 1 / (2*np.pi) # using uniform prior
        return np.trapz(p_x_v_given_s*p_s, axis=-1, x=self.s_domain) * np.trapz(p_x_a_given_s*p_s, axis=-1, x=self.s_domain)

    def likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        """
        Compute the likelihood of the separate causes hypothesis (signals from different sources).
        p(x_v, x_a|C=2) = \int \int p(x_v, x_a|s_v, s_a)p(s_v)p(s_a)ds_v ds_a and due to independence
                        = (\int p(x_v|s_v)p(s_v) ds_v)(\int p(x_a|s_a)p(s_a) ds_a)
        Currently, only uniform priors are supported, hence p(x_v, x_a|C=2) = \frac{1}{2\pi}^2

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.

        Returns:
        (float or np.ndarray): Likelihood of the separate causes hypothesis.
        """
        if (mu_p is not None) or (sigma_p is not None):
            raise NotImplementedError("Von Mises separate cause likelihood only implemented for uniform prior")
        if self.simulate:
            return self.sim_likelihood_separate_causes(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        print('Computing p(x_V, x_A| C=2) using analytic solution on sampled x_V, x_A')
        return (1/(2*np.pi)**2) * np.ones_like(x_v)

    def posterior_prob_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        """
        Compute the posterior probability of the common cause hypothesis.
        By Bayes rule P(C=1|x_v, x_a) = \frac{P(x_v, x_a| C=1) p(C=1)}{P(x_v, x_a)}
        P(C=1|x_v, x_a) = \frac{P(x_v, x_a| C=1) p(C=1)}{P(x_v, x_a|C=1)P(C=1) + P(x_v, x_a|C=2)P(C=2)}
        Here P(C=1) = p_common = pi_c.

        Parameters:
        x_v (float or np.ndarray): Visual sensory input (observed rate).
        x_a (float or np.ndarray): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float or np.ndarray): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.
        pi_c (float or np.ndarray): Prior probability of the common cause hypothesis.

        Returns:
        float or np.ndarray: Posterior probability of the common cause hypothesis.
        """
        posterior_p_common = self.likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * pi_c
        posterior_p_separate = self.likelihood_separate_causes(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * (1 - pi_c)
        return posterior_p_common / (posterior_p_common + posterior_p_separate)

    def bayesian_causal_inference(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        """
        Compute the Bayesian causal inference estimate.

        Parameters:
        x_v (float): Visual sensory input (observed rate).
        x_a (float): Auditory sensory input (observed rate).
        sigma_v (float): Visual sensory noise (concentration).
        sigma_a (float): Auditory sensory noise (concentration).
        mu_p (float): Prior mean of the stimulus rate.
        sigma_p (float): Prior noise of the stimulus rate.
        pi_c (float): Prior probability of the common cause hypothesis.

        Returns:
        float: Bayesian causal inference estimate.
        """
        # P(C=1|x_v, x_a)
        posterior_p_common = self.posterior_prob_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c)
        # \hat{s_{v, C=2}}
        segregated_estimate_v = self.segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
        # \hat{s_{a, C=2}}
        segregated_estimate_a = self.segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
        # \hat{s_{v, C=1}} = \hat{s_{a, C=1}}
        fused_estimate = self.fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        s_v_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_v
        s_a_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_a
        return s_v_hat, s_a_hat

    
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
    
def generative_model():
    """
    Generate samples (as described in "Generative model" in Kording, 2007) assuming the probability 
    of a common cause follows a Bernoulli distribution.
    """
    n_means = 100
    n_samples = 1000000
    x_vs, x_as = VonMisesCausalInference().generate_samples_causal_inference(p_common=.28, n_means=n_means, n_samples=n_samples,
                                                        kappa_1=np.ones(n_means),
                                                        kappa_2=np.ones(n_means))
    utils.plot_2d_histogram(x_vs, x_as)

def compare_likelihoods(likelihood, sim_likelihood, plot=False, title=None):
    diff_likelihood=likelihood-sim_likelihood
    print(f'Max difference between analytic and simulated likelihood: {np.max(np.abs(diff_likelihood))}')
    print(f'Max analytic and simulated likelihood: {np.max(likelihood), np.max(sim_likelihood)}')
    if plot:
        plt.hist(likelihood[:, 1,1], bins=20, label='analytic', alpha=0.5, edgecolor='b', histtype='step', density=True)
        plt.hist(sim_likelihood[:, 1, 1], bins=20, label='sim', alpha=0.5, edgecolor='r', histtype='step', density=True)
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.show()


def test_likelihoods(x_v, x_a, kappa_v, kappa_a, mu_p, kappa_p):
    model = VonMisesCausalInference()
    sim_model = VonMisesCausalInference(simulate=True)
    # compute p(x_V, x_A| C=1) by simulating x_V, x_A and using equation (4) in Kording, 2007
    likelihood_common_cause = model.likelihood_common_cause(x_v=x_v, x_a=x_a, sigma_v=kappa_v, 
                                                        sigma_a=kappa_a, mu_p=mu_p, sigma_p=kappa_p)
    # compute p(x_V, x_A| C=1) = \int p(x_V|s) p(x_A|s) p(s) ds by simulating x_V, x_A and numerical integration
    sim_likelihood_common_cause = sim_model.likelihood_common_cause(x_v=x_v, x_a=x_a, 
                                                                    sigma_v=kappa_v, sigma_a=kappa_a,
                                                                    mu_p=mu_p, sigma_p=kappa_p)
    compare_likelihoods(likelihood_common_cause, sim_likelihood_common_cause, plot=True,
                        title='Likelihood common cause (analytic vs simulation)')

    # compute p(x_V, x_A| C=2) by simulating x_V, x_A and using equation (5) in Kording, 2007
    likelihood_separate_cause = model.likelihood_separate_causes(x_v=x_v, x_a=x_a, sigma_v=kappa_v, 
                                                        sigma_a=kappa_a, mu_p=mu_p, sigma_p=kappa_p)
    # compute p(x_V, x_A| C=1) = \int p(x_V|s) p(x_A|s) p(s) ds by simulating x_V, x_A and numerical integration
    sim_likelihood_separate_cause = sim_model.likelihood_separate_causes(x_v=x_v, x_a=x_a, 
                                                                    sigma_v=kappa_v, sigma_a=kappa_a,
                                                                    mu_p=mu_p, sigma_p=kappa_p)
    compare_likelihoods(likelihood_separate_cause, sim_likelihood_separate_cause)
    del model
    del sim_model


if __name__ == "__main__":
    num_sim = 1000
    stimuli_values = np.linspace(-np.pi, np.pi, 5) 
    # Parameters for the von Mises distributions
    kappa_v, kappa_a = 300, 280  # Concentration parameters for visual and auditory inputs
    mu_p, kappa_p = None, None  # Uniform stimulus prior
    pi_c = 0.23  # Prior probability of the common cause hypothesis
    s_vs, s_as = np.meshgrid(stimuli_values, stimuli_values, indexing='ij')
    # Generate random samples for each combination of cues
    x_v = utils.wrap(vonmises.rvs(kappa=kappa_v, loc=s_vs, size=(num_sim, stimuli_values.size, stimuli_values.size)))
    x_a = utils.wrap(vonmises.rvs(kappa=kappa_a, loc=s_as, size=(num_sim, stimuli_values.size, stimuli_values.size)))
    print(f'Svs = {s_vs.reshape(-1)}\nSas = {s_as.reshape(-1)}\n')
    #test_likelihoods(x_v, x_a, kappa_v, kappa_a, mu_p, kappa_p)
    # Causal inference loop:
    model = VonMisesCausalInference()
    # Compute the posterior estimates by simulation (find \hat{s_v}=\hat{s_a} for all sample pairs (x_v, x_a))
    fused_est = utils.wrap(model.fusion_estimate(x_v, x_a, kappa_v, kappa_a, mu_p, kappa_p))
    # Compute the posterior estimates using the VM approx
    fused_est_mu, fused_est_kappa = model.fusion_posterior_params(s_v=s_vs, s_a=s_as, 
                                                                sigma_v=kappa_v, sigma_a=kappa_a, 
                                                                mu_p=mu_p, sigma_p=kappa_p)
    # Generate analytic samples from the fusion posterior distribution.
    fused_est_approx = utils.wrap(vonmises.rvs(kappa=fused_est_kappa, loc=fused_est_mu,
                                    size=(num_sim, stimuli_values.size, stimuli_values.size)))

    # Plot histograms for comparison
    v_idx, a_idx = 0, 2
    plt.hist(fused_est_approx[:, v_idx, a_idx], bins=65, label='approx', alpha=0.5, edgecolor='b', density=True, histtype='step')
    plt.hist(fused_est[:, v_idx, a_idx], bins=65, label='numeric', alpha=0.5, edgecolor='k', density=True, histtype='step')
    # plt.hist(x_v[:, v_idx, a_idx], bins=65, label='x_v', alpha=0.5, edgecolor='g', density=True, histtype='step')
    # plt.hist(x_a[:, v_idx, a_idx], bins=65, label='x_a', alpha=0.5, edgecolor='r', density=True, histtype='step')
    plt.legend()
    plt.title(f'Von Mises approximation and simulated distribution of mean responses s_v, s_a={stimuli_values[v_idx], stimuli_values[a_idx]}')
    plt.show()
    segregated_est_v = model.segregation_estimate(x=x_v, mu_p=mu_p, sigma=kappa_v, sigma_p=kappa_p)
    # TODO: check concentration for distribution of means in Von Mises 
    # https://en.wikipedia.org/wiki/Von_Mises_distribution#Distribution_of_the_mean
    segregated_est_v_analytic = vonmises.rvs(kappa=kappa_v, loc=s_vs, size=(num_sim, stimuli_values.size, stimuli_values.size))
    plt.hist(segregated_est_v_analytic[:, 1, 1], bins=20, label='analytic', alpha=0.5, edgecolor='b', histtype='step', density=True)
    plt.hist(segregated_est_v[:, 1, 1], bins=20, label='simulation', alpha=0.5, edgecolor='r', histtype='step', density=True)
    plt.legend()
    plt.title('Visual optimal estimate analytic and simulated distributions of mean responses')
    plt.show()
    segregated_est_a = model.segregation_estimate(x=x_a, mu_p=mu_p, sigma=kappa_a, sigma_p=kappa_p)
    causal_inference_est_v, causal_inference_est_a = model.bayesian_causal_inference(x_v, x_a, kappa_v,
                                                                    kappa_a, mu_p, kappa_p, pi_c)
    causal_inference.plot_histograms(causal_inference_est_v, causal_inference_est_a, x='Aud', y='Vis', 
                    x_min=stimuli_values[0], x_max=stimuli_values[-1], filepath='./fig2c_VM.png')

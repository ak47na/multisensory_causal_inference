import causal_inference
import numpy as np
from scipy.stats import vonmises, circmean
import distributions


class CustomCausalInference(causal_inference.CausalInference):
    def __init__(self, decision_rule='mean', simulate=False):
        super().__init__(distribution='vonMises', decision_rule=decision_rule)
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
        # concentration doesn't change uder our assumptions, but note the dist is not VM
        mu_c = distributions.UVM(loc=mu_c, kappa=kappa_c).decision_rule(self.decision_rule)
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
        if self.decision_rule == 'mean':
            mu_c = distributions.UVM(loc=x, kappa=sigma).mean()
        else:
            assert (self.decision_rule == 'mode')
            mu_c = distributions.UVM(loc=x, kappa=sigma).mode()
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
        # TODO(ak47na): update the pdf and test
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
        # TODO(ak47na): can we learn this?
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
        # TODO(ak47na): same
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
        # TODO(ak47na): test this doesn't change and we can inherit VMCausalInf
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
        # TODO(ak47na): test this doesn't change and we can inherit VMCausalInf
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
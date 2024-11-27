import numpy as np
from von_mises_causal_inference import VonMisesCausalInference, get_cue_combined_mean_params
import distributions
from utils import wrap
import jax.numpy as jnp


class CustomCausalInference(VonMisesCausalInference):
    def __init__(self, interp=None, decision_rule='mean', simulate=False):
        super().__init__(decision_rule=decision_rule)
        self.simulate = simulate
        self.interp = interp
        self.s_domain = jnp.linspace(-jnp.pi, jnp.pi, 1000).reshape(1, 1, 1, -1)

    def fusion_estimate(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, simulate=False, return_sigma=False):
        """
        Compute the MAP* estimate in the fusion case (combined sensory processing).
        The posterior p(s|x_v, x_a) \propto p(x_v, x_a|s)p(s) = p(x_a|s)p(x_v|s)p(s) with p(s) as the
        prior and p(x_a|s), p(x_v|s) as the likelihoods.
        The prior is currently assumed uniform and its parameters are not used.
        The decision_rule (mean or mode) is applied after converting back to angle
        space on U^{-1}(p(s|x_v, x_a)).

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
        # Map the posterior distribution VM(mu_c, kappa_c) to angle space
        # Concentration doesn't change under our assumptions, but note the distribution is not VM.
        mu_c = distributions.UVM(loc=mu_c, kappa=kappa_c, scale=None, 
                                 interp=self.interp).decision_rule(self.decision_rule)
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
        Compute the MAP* estimate in the segregation case (independent sensory processing).
        The posterior p(s|x) \propto p(x|s)p(s), with p(s) as the prior and p(x|s) as the likelihood.
        For uniform p(s), the decision_rule (mean or mode) is applied after converting back to angle
        space on U^{-1}(p(s|x)).

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
        return distributions.UVM(loc=x, kappa=sigma, scale=None, 
                                 interp=self.interp).decision_rule(self.decision_rule)

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
        # P(C=1|x_v, x_a) (unchanged by the uniformising map as the integration is in internal space)
        posterior_p_common = self.posterior_prob_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c)
        # \hat{s_{v, C=2}}
        segregated_estimate_v = self.segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
        # \hat{s_{a, C=2}}
        segregated_estimate_a = self.segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
        # \hat{s_{v, C=1}} = \hat{s_{a, C=1}}
        fused_estimate = self.fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        s_v_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_v
        s_a_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_a
        return wrap(s_v_hat), wrap(s_a_hat)

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import causal_inference


class GaussianCausalInference(causal_inference.CausalInference):
    def __init__(self, distribution='gaussian', simulate=False):
        super().__init__(distribution)
        self.simulate = simulate
        self.s_domain = np.linspace(-10, 10, 1000).reshape(1, 1, 1, -1)

    def segregation_estimate(self, x, mu_p, sigma, sigma_p):
        return (x * sigma_p**2 + mu_p * sigma**2) / (sigma**2 + sigma_p**2)
    
    def sim_likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        # res = \int_{s in s_domain} N(x_a; s, sigma_a) * N(x_a; s, sigma_a) * p(s)
        p_x_v_given_s = norm.pdf(x=x_a[..., np.newaxis], loc=self.s_domain, scale=sigma_a)
        p_x_a_given_s = norm.pdf(x=x_v[..., np.newaxis], loc=self.s_domain, scale=sigma_v)
        p_s = norm.pdf(x=self.s_domain, loc=mu_p, scale=sigma_p)
        return np.trapz(p_x_v_given_s*p_x_a_given_s*p_s, axis=-1, x=self.s_domain)

    def fusion_estimate(self, x_a, x_v, sigma_a, sigma_v, mu_p, sigma_p, return_sigma=False):
        num = x_a * ((sigma_v*sigma_p)**2) + x_v * ((sigma_a*sigma_p)**2) + mu_p * ((sigma_v*sigma_a)**2)
        denom = ((sigma_v*sigma_p)**2) + ((sigma_a*sigma_p)**2) + ((sigma_v*sigma_a)**2)
        if return_sigma:
            return num / denom, 1/np.sqrt((1/sigma_a**2)+(1/sigma_v**2)+(1/sigma_p**2))
        return num / denom
    
    def fusion_posterior_params(self, s_a, s_v, sigma_a, sigma_v, mu_p, sigma_p):
        fused_est_mu, fused_est_sigma  = self.fusion_estimate(x_a=s_a, x_v=s_v, 
                                                            sigma_a=sigma_a, 
                                                            sigma_v=sigma_v, 
                                                            mu_p=mu_p, 
                                                            sigma_p=sigma_p,
                                                            return_sigma=True)
        return fused_est_mu, fused_est_sigma

    def likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        if self.simulate:
            return self.sim_likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        var_common = sigma_v**2 * sigma_a**2 + sigma_v**2 * sigma_p**2 + sigma_a**2 * sigma_p**2
        exp_common = ((x_v - x_a)**2 * sigma_p**2 + (x_v - mu_p)**2 * sigma_a**2 + (x_a - mu_p)**2 * sigma_v**2) / (2 * var_common)
        return np.exp(-exp_common) / (2 * np.pi * np.sqrt(var_common))

    def likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        var_sep_v = sigma_v**2 + sigma_p**2
        var_sep_a = sigma_a**2 + sigma_p**2
        exp_sep = ((x_v - mu_p)**2 / (2 * var_sep_v)) + ((x_a - mu_p)**2 / (2 * var_sep_a))
        return np.exp(-exp_sep) / (2 * np.pi * np.sqrt(var_sep_v * var_sep_a))

    def posterior_prob_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        posterior_p_common = self.likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * pi_c
        posterior_p_separate = self.likelihood_separate_causes(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * (1 - pi_c)
        return posterior_p_common / (posterior_p_common + posterior_p_separate)

    def bayesian_causal_inference(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        posterior_p_common = self.posterior_prob_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c)
        segregated_estimate_v = self.segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
        segregated_estimate_a = self.segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
        fused_estimate = self.fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
        s_v_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_v
        s_a_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_a
        return s_v_hat, s_a_hat


if __name__ == "__main__()":
    # Data analysis figure2: "we simulate the system for each combination of cues for 10,000 times"
    num_sim = 10000
    model = GaussianCausalInference()
    stimuli_values = np.linspace(-10, 10, 5)
    s_vs, s_as = np.meshgrid(stimuli_values, stimuli_values, indexing='ij')

    sigma_v, sigma_a = 2.14, 9.2  # Sensory noise for visual and auditory inputs (9.2+-1.1)
    mu_p, sigma_p = 0, 12.3  # Prior mean and standard deviation for the stimulus rate
    pi_c = 0.23  # Prior probability of the common cause hypothesis
    print(f'Svs = {s_vs.reshape(-1)}\nSas = {s_as.reshape(-1)}\n')

    # Generate random samples for each combination of cues
    x_v = norm.rvs(loc=s_vs, scale=sigma_v, size=(num_sim, stimuli_values.size, stimuli_values.size))
    x_a = norm.rvs(loc=s_as, scale=sigma_a, size=(num_sim, stimuli_values.size, stimuli_values.size))

    # Compute the estimates using the defined functions
    segregated_est_v = model.segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
    segregated_est_a = model.segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
    fused_est = model.fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
    causal_inference_est_v, causal_inference_est_a = model.bayesian_causal_inference(x_v, x_a, sigma_v,
                                                                    sigma_a, mu_p, sigma_p, pi_c)

    print(f'segregated_est_v={segregated_est_v.mean(axis=0)}\n')
    print(f'segregated_est_a={segregated_est_a.mean(axis=0)}\n')
    print(f'fused_est={fused_est.mean(axis=0)}\n')
    print(f'causal_inference_est_v={causal_inference_est_v.mean(axis=0)}\n')
    print(f'causal_inference_est_a={causal_inference_est_a.mean(axis=0)}\n')

    causal_inference.plot_histograms(causal_inference_est_v, causal_inference_est_a, x='Aud', y='Vis', 
                    x_min=stimuli_values[0], x_max=stimuli_values[-1], filepath='./fig2c.png')

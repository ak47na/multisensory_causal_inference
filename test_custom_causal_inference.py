import unittest
import numpy as np
from custom_causal_inference import CustomCausalInference
from von_mises_causal_inference import get_cue_combined_mean_params
import distributions
import forward_models_causal_inference


class TestCustomCausalInference(unittest.TestCase):

    def setUp(self):
        self.decision_rules = ['mean', 'mode']
        self.models = [CustomCausalInference(decision_rule='mean'), 
                       CustomCausalInference(decision_rule='mode')]
        self.interp = distributions.get_interp()
        self.error_delta = 1e-5
 
    def test_fusion_estimate(self):
        # Use simple fixed inputs
        x_v = np.array([1.0])
        x_a = np.array([1.5])
        sigma_v = 2.0
        sigma_a = 3.0

        # No prior for Von Mises distributions
        mu_p = None
        sigma_p = None

        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_v, mu2=x_a, kappa1=sigma_v, 
                                                     kappa2=sigma_a)
    
        for i, decision_rule in enumerate(self.decision_rules):
            combined_dist = distributions.UVM(loc=mu_c, kappa=kappa_c, scale=None, 
                                        interp=self.interp)
            expected_result = combined_dist.decision_rule(decision_rule)
            result = self.models[i].fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
            self.assertAlmostEqual(result, expected_result, delta=self.error_delta)
            # Test the NotImplementedError with non-uniform priors
            with self.assertRaises(NotImplementedError):
                self.models[i].fusion_estimate(x_v, x_a, sigma_v, sigma_a, 0.5, 1.0)
    
    def test_fusion_estimate_variable_kappas(self):
        x_v = np.array([1.0, 0.5]).reshape((2, 1))
        x_a = np.array([1.5, -0.5]).reshape((2, 1))
        kappa1 = forward_models_causal_inference.reshape_kappa_for_causal_inference(np.array([1.5, 2.0, 3.0]), num_mus=2)
        kappa2 = forward_models_causal_inference.reshape_kappa_for_causal_inference(np.array([1.5, 2.3, .5]), num_mus=2)
        # No prior for Von Mises distributions
        mu_p = None
        sigma_p = None

        # Compute the fusion estimate for all [(x_v, kappa1); (x_a, kappa2)] pairs
        mu_c, kappa_c = get_cue_combined_mean_params(mu1=x_v, mu2=x_a, kappa1=kappa1, 
                                                     kappa2=kappa2)
    
        for i, decision_rule in enumerate(self.decision_rules):
            result = self.models[i].fusion_estimate(x_v, x_a, kappa1, kappa2, mu_p, sigma_p)
            for x_index, _x_v in enumerate(x_v):
                for k_index, _kappa1 in enumerate(kappa1[0]):
                    combined_dist = distributions.UVM(loc=mu_c[x_index, k_index:k_index+1], 
                                                      kappa=kappa_c[x_index, k_index], 
                                                      scale=None, 
                                                      interp=self.interp)
                    expected_result = combined_dist.decision_rule(decision_rule)
                    self.assertAlmostEqual(result[x_index, k_index], expected_result, delta=self.error_delta)
            # Test the NotImplementedError with non-uniform priors
            with self.assertRaises(NotImplementedError):
                self.models[i].fusion_estimate(x_v, x_a, kappa1, kappa2, 0.5, 1.0)

    def test_fusion_posterior_params(self):
        s_v = np.array([1.0])
        s_a = np.array([1.5])
        sigma_v = 2.0
        sigma_a = 3.0

        mu_p = None
        sigma_p = None
        # fused_sigma == kappa_c == np.sqrt(4+9+12*np.cos(.5))
        mu_c, kappa_c = np.array([1.30102406]), np.array([4.85087526])
        uvm = distributions.UVM(loc=mu_c, scale=None, kappa=kappa_c, interp=self.interp)
        for i, decision_rule in enumerate(self.decision_rules):
            fused_mu, fused_sigma = self.models[i].fusion_posterior_params(s_v, s_a, sigma_v, sigma_a,
                                                                            mu_p, sigma_p)
            expected_fused_mu, expected_fused_sigma = uvm.decision_rule(decision_rule), kappa_c
            self.assertAlmostEqual(fused_mu, expected_fused_mu, delta=self.error_delta)
            self.assertAlmostEqual(fused_sigma, expected_fused_sigma, delta=self.error_delta)

    def test_segregation_estimate(self):
        x = np.array([1.0])
        kappa =  np.array([2.0])
        mu_p = None
        sigma_p = None
        uvm_x = distributions.UVM(loc=x, scale=None, kappa=kappa, interp=self.interp)

        for i, decision_rule in enumerate(self.decision_rules):
            result = self.models[i].segregation_estimate(x=x, mu_p=mu_p, sigma=kappa, sigma_p=sigma_p)
            expected_result = uvm_x.decision_rule(decision_rule)  # Expectation in this simplified case
            self.assertAlmostEqual(result, expected_result, delta=self.error_delta)

            with self.assertRaises(NotImplementedError):
                self.models[i].segregation_estimate(x=x, mu_p=0.5, sigma=kappa, sigma_p=sigma_p)

    def test_segregation_estimate_variable_kappa(self):    
        x = np.array([1.0, 0.5]).reshape((2, 1))
        kappa = forward_models_causal_inference.reshape_kappa_for_causal_inference(np.array([1.5, 2.0, 3.0]), num_mus=2)
        mu_p = None
        sigma_p = None

        for i, decision_rule in enumerate(self.decision_rules):
            result = self.models[i].segregation_estimate(x=x, mu_p=mu_p, sigma=kappa, sigma_p=sigma_p)
            for x_index, _x in enumerate(x):
                for kappa_index, _kappa in enumerate(kappa[0]):
                    uvm__x = distributions.UVM(loc=_x, scale=None, kappa=_kappa, interp=self.interp)
                    expected_result = uvm__x.decision_rule(decision_rule)  # Expectation in this simplified case
                    self.assertAlmostEqual(result[x_index, kappa_index], expected_result, delta=self.error_delta)

            with self.assertRaises(NotImplementedError):
                self.models[i].segregation_estimate(x=x, mu_p=0.5, sigma=kappa, sigma_p=sigma_p)

    def test_bayesian_causal_inference(self):
        s_v = np.array([1.0])
        x_v_resp = np.array([[0.69186568], [0.31415922]]) # mean and mode responses
        s_a = np.array([1.5])
        x_a_resp = np.array([[1.28445547], [0.41469019]]) # mean and mode responses
        fusion_resp = np.array([[0.99348235], [0.46495567]]) # mean and mode responses
        sigma_v = 2.0
        sigma_a = 3.0
        pi_c = 0.8
        mu_p = None
        sigma_p = None

        for i, _ in enumerate(self.decision_rules):
            s_v_hat, s_a_hat = self.models[i].bayesian_causal_inference(x_v=s_v, x_a=s_a, 
                                                                       sigma_v=sigma_v, 
                                                                       sigma_a=sigma_a, 
                                                                       mu_p=mu_p, 
                                                                       sigma_p=sigma_p, 
                                                                       pi_c=pi_c)
            post_pi_c = self.models[i].posterior_prob_common_cause(x_v=s_v, x_a=s_a, 
                                                                  sigma_v=sigma_v, 
                                                                  sigma_a=sigma_a, 
                                                                  mu_p=mu_p, 
                                                                  sigma_p=sigma_p, 
                                                                  pi_c=pi_c)

            expected_s_v_hat = (post_pi_c * fusion_resp[i] + (1 - post_pi_c) * x_v_resp[i])
            expected_s_a_hat = (post_pi_c * fusion_resp[i] + (1 - post_pi_c) * x_a_resp[i])
            self.assertAlmostEqual(s_v_hat, expected_s_v_hat, delta=self.error_delta)
            self.assertAlmostEqual(s_a_hat, expected_s_a_hat, delta=self.error_delta)

if __name__ == '__main__':
    unittest.main()

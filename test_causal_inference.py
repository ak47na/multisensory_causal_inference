import unittest
import numpy as np
from custom_causal_inference import CustomCausalInference
import von_mises_causal_inference 
import distributions
import forward_models_causal_inference
import utils


class TestCausalInference(unittest.TestCase):
    def test_cue_combined_mean_params(self):
        x_v = np.array([1.0, 0.5]).reshape((2, 1))
        x_a = np.array([1.5, -0.5]).reshape((2, 1))
        kappa1 = forward_models_causal_inference.reshape_kappa_for_causal_inference(np.array([1.5, 2.0, 3.0]), num_mus=2)
        kappa2 = forward_models_causal_inference.reshape_kappa_for_causal_inference(np.array([1.5, 2.3, .5]), num_mus=2)
        # Compute the fusion estimate for all 4 [(x_v, kappa1); (x_a, kappa2)] pairs
        mu_c, kappa_c = von_mises_causal_inference.get_cue_combined_mean_params(mu1=x_v, mu2=x_a, kappa1=kappa1, 
                                                        kappa2=kappa2)
        for i in range(mu_c.shape[0]):
            for j in range(mu_c.shape[1]):
                self.assertAlmostEqual(mu_c[i,j], utils.wrap(x_a[i] + np.arctan2(np.sin(x_v[i]-x_a[i]), 
                                                                    kappa2[i,j]/kappa1[i,j] + np.cos(x_v[i]-x_a[i]))))
                self.assertAlmostEqual(kappa_c[i,j], np.sqrt((kappa1[i,j]**2) + (kappa2[i,j]**2) + 2*kappa1[i,j]*kappa2[i,j]*np.cos(x_v[i]-x_a[i])))


if __name__ == '__main__':
    unittest.main()
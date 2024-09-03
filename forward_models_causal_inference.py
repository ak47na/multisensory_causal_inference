import numpy as np
import pickle
import uniformised_space_utils as usu
from scipy.stats import vonmises, circmean
import matplotlib.pyplot as plt
from custom_causal_inference import CustomCausalInference
from repulsion_hypothesis import repulsion_value
import utils


def reshape_kappa_for_sampling(kappa):
    if isinstance(kappa, (int, float)):
        # Fixed concentration for all means
        return np.array([[kappa]])
    elif kappa.ndim == 1:
        # The same possible concentration values for all means, reshape to (1, len(kappa))
        return kappa[np.newaxis, :]
    else:
        # Different possible concentrations for means, shape must be (len(mus), len(kappas))
        assert (kappa.ndim == 2)
    return kappa


def reshape_kappa_for_causal_inference(kappa, num_mus):
    if kappa.ndim == 2:
        assert (kappa.shape[0] == num_mus)
    kappa = reshape_kappa_for_sampling(kappa)
    kappa = np.tile(kappa, reps=(num_mus, 1))
    assert (kappa.ndim == 2) and (kappa.shape[0] == num_mus)
    return kappa

class CausalEstimator:
    def __init__(self, model, angle_gam_data_path, unif_fn_data_path, mu_p=None, sigma_p=None, num_sim=10000):
        self.model = model
        # Load the GAM.
        with open(angle_gam_data_path, 'rb') as file:
            self.gam_data = pickle.load(file)
        # Load the uniformising function data.
        with open(unif_fn_data_path, 'rb') as file:
            unif_fn_data = pickle.load(file)
        # Initialise uniformising function map.
        self.unif_map = usu.UnifMap(data=unif_fn_data)
        self.unif_map.get_cdf_and_inverse_cdf()
        self.mu_p=mu_p
        self.sigma_p=sigma_p
        self.num_sim = num_sim
        self.grid = np.linspace(-np.pi, np.pi, num=250)

    def get_vm_samples(self, num_sim, mu_t, mu_s_n, kappa1, kappa2):
        """
        Generate samples from Von Mises distributions for t and s_n.

        Parameters:
        num_sim (int): Number of simulations to run (i.e., the number of samples to generate).
        mu_t (ndarray|int|float): Mean directions for the distributions of regressor t.
        mu_s_n (ndarray|int|float): Mean directions for the distributions of s_n.
        kappa1 (ndarray|int|float): Concentration parameters for the distributions of regressor t.
        kappa2 (ndarray|int|float): Concentration parameters for the distributions of s_n.

        Returns:
        tuple: Two ndarrays representing internal samples for the two cues:
            - t_samples (ndarray): Samples from the t distribution, shape (num_sim, len(mu_t), kappa1.shape[1]).
            - s_n_samples (ndarray): Samples from the s_n distribution, shape (num_sim, len(mu_s_n), kappa2.shape[1]).
        """
        kappa1 = reshape_kappa_for_sampling(kappa1)
        kappa2 = reshape_kappa_for_sampling(kappa2)
        t_samples = vonmises(loc=mu_t[:, np.newaxis], kappa=kappa1).rvs(size=(num_sim, mu_t.shape[0], kappa1.shape[1]))
        s_n_samples = vonmises(loc=mu_s_n[:, np.newaxis], kappa=kappa2).rvs(size=(num_sim, mu_s_n.shape[0], kappa2.shape[1]))
        return t_samples, s_n_samples
    
    def forward_from_means(self, mu_t, mu_s_n, kappa1, kappa2, p_common, num_sim=None):
        """
        Generate samples from Von Mises distributions and pass them through the forward model.

        Parameters:
        mu_t (ndarray|int|float): Mean directions for the distributions of regressor t.
        mu_s_n (ndarray|int|float): Mean directions for the distributions of s_n.
        kappa1 (ndarray|int|float): Concentration parameters for the distributions of regressor t.
        kappa2 (ndarray|int|float): Concentration parameters for the distributions of s_n.
        p_common (float): Prior probability of a common cause.
        num_sim (int, optional): Number of simulations to run. If None, use self.num_sim.

        Returns:
        tuple: Results from the forward model, including responses, posterior probability, 
                and circular mean estimates for t and s_n.
        """
        if num_sim is None:
            num_sim = self.num_sim
        t_samples, s_n_samples = self.get_vm_samples(num_sim=num_sim, mu_t=mu_t, mu_s_n=mu_s_n,
                                                     kappa1=kappa1, kappa2=kappa2)
        return self.forward(t_samples=t_samples, s_n_samples=s_n_samples, kappa1=kappa1, 
                            kappa2=kappa2, p_common=p_common)

    def forward(self, t_samples, s_n_samples, kappa1, kappa2, p_common):
        """
        Run the forward model to find optimal estimates and posteriors of the common cause probability.

        Parameters:
        t_samples (ndarray): Samples from the t distribution, shape (num_sim, N, K).
        s_n_samples (ndarray): Samples from the s_n distribution, shape (num_sim, N, K).
        kappa1 (ndarray|int|float): Concentration parameters for the distributions of regressor t.
        kappa2 (ndarray|int|float): Concentration parameters for the distributions of s_n.
        p_common (float): Prior probability of a common cause.

        Returns:
        tuple: 
            - responses: Tuple containing optimal estimates for t and s_n.
            - posterior_p_common (ndarray): Posterior probabilities of a common cause, shape (num_sim, N, K).
            - mean_t_est (ndarray): Circular mean estimate for t, shape (N, K).
            - mean_sn_est (ndarray): Circular mean estimate for s_n, shape (N, K).
        """
        kappa1 = reshape_kappa_for_causal_inference(kappa1, num_mus=t_samples.shape[1])
        kappa2 = reshape_kappa_for_causal_inference(kappa2, num_mus=s_n_samples.shape[1])
        
        responses = (self.model.bayesian_causal_inference(x_v=t_samples, 
                                                        x_a=s_n_samples, 
                                                        sigma_v=kappa1, 
                                                        sigma_a=kappa2,
                                                        mu_p=self.mu_p, 
                                                        sigma_p=self.sigma_p,
                                                        pi_c=p_common))
        posterior_p_common = self.model.posterior_prob_common_cause(x_v=t_samples, 
                                                                    x_a=s_n_samples, 
                                                                    sigma_v=kappa1, 
                                                                    sigma_a=kappa2,
                                                                    mu_p=self.mu_p, 
                                                                    sigma_p=self.sigma_p,
                                                                    pi_c=p_common)
        # Find circular mean across "optimal" estimates for samples.
        mean_t_est = circmean(self.unif_map.unif_space_to_angle_space(responses[0]), 
                                low=-np.pi, high=np.pi, axis=0)
        mean_sn_est = circmean(self.unif_map.unif_space_to_angle_space(responses[1]),
                                low=-np.pi, high=np.pi, axis=0)
        return responses, posterior_p_common, mean_t_est, mean_sn_est
    

if __name__ == "__main__":
    causal_inference_estimator = CausalEstimator(model=CustomCausalInference(decision_rule='mean'),
                                                 angle_gam_data_path='D:/AK_Q1_2024/Gatsby/data/base_bayesian_contour_1_circular_gam/base_bayesian_contour_1_circular_gam.pkl',
                                                 unif_fn_data_path='D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl')
    p_commons = [0, .2, .5, .7, 1]
    results = {
        'responses': [], 
        'posterior_p_common': [],
        'mean_t_est': [], 
        'mean_sn_est': []
    }
    s_n, t, r_n = utils.get_s_n_and_t(causal_inference_estimator.grid, 
                                   causal_inference_estimator.gam_data,
                                   step=200)
    
    ut = causal_inference_estimator.unif_map.angle_space_to_unif_space(t.reshape(-1))
    us_n = causal_inference_estimator.unif_map.angle_space_to_unif_space(s_n.reshape(-1))
    print(f'Performing causal inference for ut, u_s_n of shape {ut.shape, us_n.shape}')

    # kappa1 = np.tile(np.array([25, 50, 75, 100, 125, 150]), reps=(len(ut), 1))
    # kappa2 = np.tile(np.array([25, 50, 75, 100, 125, 150]), reps=(len(us_n), 1))
    kappa1 = np.array([25, 100])
    kappa2 = np.array([25, 100])

    # Generate samples for the simulation
    results['num_sim'] = 10000
    t_samples, s_n_samples = causal_inference_estimator.get_vm_samples(num_sim=results['num_sim'], 
                                                                       mu_t=ut, mu_s_n=us_n,
                                                                       kappa1=kappa1, 
                                                                       kappa2=kappa2)
    results['ut'] = ut
    results['us_n'] = us_n
    for p_common in p_commons:
        responses, posterior_p_common, mean_t_est, mean_sn_est = causal_inference_estimator.forward(t_samples=t_samples,
                                                                                                    s_n_samples=s_n_samples,
                                                                                                    p_common=p_common,
                                                                                                    kappa1=kappa1,
                                                                                                    kappa2=kappa2)
        results['responses'].append(responses)
        results['posterior_p_common'].append(posterior_p_common)
        results['mean_t_est'].append(mean_t_est)
        results['mean_sn_est'].append(mean_sn_est)

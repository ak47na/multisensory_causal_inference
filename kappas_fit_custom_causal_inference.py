import multiprocessing as mp
import os
import numpy as np
from scipy.stats import vonmises, circmean
from custom_causal_inference import CustomCausalInference
from repulsion_hypothesis import repulsion_value
import utils
import plots
import forward_models_causal_inference
import matplotlib.pyplot as plt

def compute_error(computed_values, data_slice):
    return utils.circular_dist(computed_values, data_slice)

def process_mean_pair(args):
    mean_indices, ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, angle_gam_data_path, unif_fn_data_path = args

    mu1 = ut[mean_indices]
    mu2 = us_n[mean_indices]
    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    print("ID of process running : {}".format(os.getpid()), 'mean shapes', mu1.shape, mu2.shape)

    # import pdb; pdb.set_trace()
    # TODO: generate new seed every time
    t_samples, s_n_samples = causal_inference_estimator.get_vm_samples(num_sim=num_sim, 
                                                                    mu_t=mu1, mu_s_n=mu2,
                                                                    kappa1=kappa1_flat, 
                                                                    kappa2=kappa2_flat)
    responses, posterior_p_common, mean_t_est, mean_sn_est = causal_inference_estimator.forward(t_samples=t_samples,
                                                                                                s_n_samples=s_n_samples,
                                                                                                p_common=p_common,
                                                                                                kappa1=kappa1_flat,
                                                                                                kappa2=kappa2_flat)
    del causal_inference_estimator
    del t_samples, s_n_samples
    errors = compute_error(mean_sn_est, data_slice)
    assert mean_sn_est.ndim == 2
    print(f'mean_sn_est_shape={mean_sn_est.shape}, errors_shape={errors.shape}')
    assert mean_sn_est.ndim == 2, f'Found mean_sn_est_shape={mean_sn_est.shape}'
    # argmin flattens the array, unravel_index finds the argmin index coordinates in the 2D grid of kappas
    idx_min = np.argmin(errors, axis=1)
    optimal_kappa1 = kappa1_flat[idx_min]
    optimal_kappa2 = kappa2_flat[idx_min]
    min_error = np.min(errors, axis=1)
    # TODO: uncomment line below or save results?
    #TODO: fix axis computation, likely need 2D for means
    # return (responses, posterior_p_common, mean_t_est, i, j, (optimal_kappa1, optimal_kappa2), min_error)
    return (mean_indices, p_common, (optimal_kappa1, optimal_kappa2), min_error)

def find_optimal_kappas(grid_dim):
    tasks = []
    print(f'Fitting for grid_dim={grid_dim}, num_means={ut.shape}, data_shape={r_n.shape}')
    
    for i, data_slice in enumerate(r_n):
        for p_common in p_commons:
            tasks.append((np.array([i]), ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, angle_gam_data_path, unif_fn_data_path))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_mean_pair, tasks)

    optimal_kappa_pairs = {}
    min_error_for_idx = {idx: np.pi for idx in range(r_n.shape[0])}
    for mean_indices, p_common, optimal_kappa_pair, min_error in results:
        for i, idx in enumerate(mean_indices):
            optimal_kappa_pairs[(idx, p_common)] = (optimal_kappa_pair[0][i], optimal_kappa_pair[1][i])
            min_error_for_idx[idx] = min(min_error_for_idx[idx], min_error[i])
    return optimal_kappa_pairs, min_error_for_idx

if __name__ == '__main__':
    num_sim = 1000
    D = 250  # grid dimension 
    angle_gam_data_path = 'D:/AK_Q1_2024/Gatsby/data/base_bayesian_contour_1_circular_gam/base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path='D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl'
    p_commons = [0, .2, .5, .7, 1]

    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map

    s_n, t, r_n = utils.get_s_n_and_t(causal_inference_estimator.grid, 
                                   causal_inference_estimator.gam_data)
    print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')

    #Further filtering
    num_means = 2
    step=len(s_n)//num_means
    indices = np.arange(0, s_n.shape[0], step=step)
    mu_x_dim = len(indices)
    s_n = s_n[indices][:, indices]
    t = t[indices][:, indices]
    r_n = r_n[indices][:, indices]
    plots.heatmap_f_s_n_t(f_s_n_t=r_n, s_n=s_n, t=t, f_name='r_n')

    min_kappa1, max_kappa1, num_kappa1s = 1, 200, 100
    min_kappa2, max_kappa2, num_kappa2s = 1.1, 300, 100
    s_n, t, r_n = s_n.flatten(), t.flatten(), r_n.flatten()
    us_n = unif_map.angle_space_to_unif_space(s_n)
    ut = unif_map.angle_space_to_unif_space(t)
    kappa1 = np.logspace(start=np.log10(min_kappa1), stop=np.log10(max_kappa1), num=num_kappa1s, base=10)
    kappa2 = np.logspace(start=np.log10(min_kappa2), stop=np.log10(max_kappa2), num=num_kappa2s, base=10)
    kappa1_grid, kappa2_grid = np.meshgrid(kappa1, kappa2, indexing='ij')
    kappa1_flat, kappa2_flat = kappa1_grid.flatten(), kappa2_grid.flatten()
    print(f'Performing causal inference for ut, u_s_n of shape {ut.shape, us_n.shape}')
    plt.plot(kappa1, label='kappa1')
    plt.plot(kappa2, label='kappa2')
    plt.legend()
    plt.show()

    optimal_kappa_pairs, min_error_for_idx = find_optimal_kappas(grid_dim=mu_x_dim)
    print(f'completed with opt res = {optimal_kappa_pairs}')
    plt.plot(min_error_for_idx.keys(), min_error_for_idx.values())
    plt.show()
    print(f'max error = {max(min_error_for_idx.values())}, avg error: {np.mean(np.array(list(min_error_for_idx.values())))}')


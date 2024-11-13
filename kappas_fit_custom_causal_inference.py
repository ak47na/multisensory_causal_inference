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
import argparse
import pickle
from tqdm import tqdm


def compute_error(computed_values, data_slice):
    return utils.circular_dist(computed_values, data_slice)

def init_worker(angle_gam_data_path, unif_fn_data_path):
    global causal_inference_estimator
    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    np.random.seed(os.getpid())

def process_mean_pair(args):
    mean_indices, ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices = args

    mu1 = ut[mean_indices]
    mu2 = us_n[mean_indices]

    # Select the chunk of kappa combinations
    kappa1_chunk = kappa1_flat[kappa_indices]
    kappa2_chunk = kappa2_flat[kappa_indices]

    # Generate samples for the kappa chunk
    t_samples, s_n_samples = causal_inference_estimator.get_vm_samples(
        num_sim=num_sim,
        mu_t=mu1,
        mu_s_n=mu2,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)

    # Perform forward computation
    responses, posterior_p_common, mean_t_est, mean_sn_est = causal_inference_estimator.forward(
        t_samples=t_samples,
        s_n_samples=s_n_samples,
        p_common=p_common,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)
    del t_samples, s_n_samples

    errors = compute_error(mean_sn_est, data_slice)
    assert mean_sn_est.ndim == 2
    print(f'mean_sn_est_shape={mean_sn_est.shape}, errors_shape={errors.shape}')
    assert mean_sn_est.ndim == 2, f'Found mean_sn_est_shape={mean_sn_est.shape}'

    # Find the minimum error in this chunk
    idx_min = np.argmin(errors, axis=1)
    optimal_kappa1 = kappa1_chunk[idx_min]
    optimal_kappa2 = kappa2_chunk[idx_min]
    min_error = np.min(errors, axis=1)
    print("Process ID: {}, mean shapes: {}, {}, min error {}".format(os.getpid(), mu1.shape, mu2.shape, min_error))

    return (mean_indices, p_common, (optimal_kappa1, optimal_kappa2), min_error)

def find_optimal_kappas():
    tasks = []
    print(f'Fitting for num_means={ut.shape}, data_shape={r_n.shape}')

    # Define the chunk size
    chunk_size = 500  # Adjust this based on your memory constraints

    total_kappa_combinations = len(kappa1_flat)
    kappa_indices = np.arange(total_kappa_combinations)

    num_chunks = (total_kappa_combinations + chunk_size - 1) // chunk_size

    for i, data_slice in enumerate(r_n):
        for p_common in p_commons:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_kappa_combinations)
                kappa_indices_chunk = kappa_indices[start_idx:end_idx]
                tasks.append((np.array([i]), ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices_chunk))

    initargs = (angle_gam_data_path, unif_fn_data_path)
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=initargs) as pool:
        results = []
        # Use tqdm to add a progress bar
        for result in tqdm(pool.imap_unordered(process_mean_pair, tasks), total=len(tasks)):
            results.append(result)


    # Collect and combine results
    optimal_kappa_pairs = {}
    min_error_for_idx = {idx: np.pi for idx in range(r_n.shape[0])}

    # Since we have multiple results for each mean_indices (one per chunk), we need to find the overall minimum
    for mean_indices, p_common, optimal_kappa_pair, min_error in results:
        idx = mean_indices[0]  # Since mean_indices is an array of one element
        key = (idx, p_common)
        if key not in optimal_kappa_pairs or min_error < min_error_for_idx[idx]:
            optimal_kappa_pairs[key] = (optimal_kappa_pair[0], optimal_kappa_pair[1])
            min_error_for_idx[idx] = min_error

    return optimal_kappa_pairs, min_error_for_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit kappas for grid pairs as specified by arguments.")
    parser.add_argument('--use_high_cc_error_pairs', type=bool, default=False, help='True if grid pairs are selected based on cue combination errors')

    num_sim = 1000
    D = 250  # grid dimension
    angle_gam_data_path = 'D:/AK_Q1_2024/Gatsby/data/base_bayesian_contour_1_circular_gam/base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path = 'D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl'
    p_commons = np.linspace(start=0, stop=1, num=10)
    args = parser.parse_args()
    use_high_cc_error_pairs = args.use_high_cc_error_pairs

    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map

    if use_high_cc_error_pairs:
        s_n, t, r_n = utils.get_cc_high_error_pairs(causal_inference_estimator.grid,
                                        causal_inference_estimator.gam_data,
                                        max_samples=50)
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
    else:
        s_n, t, r_n = utils.get_s_n_and_t(causal_inference_estimator.grid,
                                        causal_inference_estimator.gam_data)
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')

        # Further filtering
        num_means = 2
        step = len(s_n) // num_means
        indices = np.arange(0, s_n.shape[0], step=step)
        mu_x_dim = len(indices)
        s_n = s_n[indices][:, indices]
        t = t[indices][:, indices]
        r_n = r_n[indices][:, indices]
        plots.heatmap_f_s_n_t(f_s_n_t=r_n, s_n=s_n, t=t, f_name='r_n')

    min_kappa1, max_kappa1, num_kappa1s = 1, 200, 50
    min_kappa2, max_kappa2, num_kappa2s = 1.1, 300, 50
    s_n, t, r_n = s_n.flatten(), t.flatten(), r_n.flatten()
    us_n = unif_map.angle_space_to_unif_space(s_n)
    ut = unif_map.angle_space_to_unif_space(t)
    kappa1 = np.logspace(start=np.log10(min_kappa1), stop=np.log10(max_kappa1), num=num_kappa1s, base=10)
    kappa2 = np.logspace(start=np.log10(min_kappa2), stop=np.log10(max_kappa2), num=num_kappa2s, base=10)
    kappa1_grid, kappa2_grid = np.meshgrid(kappa1, kappa2, indexing='ij')
    kappa1_flat, kappa2_flat = kappa1_grid.flatten(), kappa2_grid.flatten()
    print(f'Performing causal inference for ut, us_n of shape {ut.shape, us_n.shape}')
    plt.plot(kappa1, label='kappa1')
    plt.plot(kappa2, label='kappa2')
    plt.legend()
    plt.show()

    optimal_kappa_pairs, min_error_for_idx = find_optimal_kappas()
    print(f'Completed with optimal results = {optimal_kappa_pairs}')
    with open('./learned_data/optimal_kappa_pairs.pkl', 'wb') as f:
        pickle.dump(optimal_kappa_pairs, f)
    with open('./learned_data/min_error_for_idx.pkl', 'wb') as f:
        pickle.dump(min_error_for_idx, f)
    np.save('./learned_data/selected_s_n.npy', arr=s_n)
    np.save('./learned_data/selected_t.npy', arr=t)
    np.save('./learned_data/selected_r_n.npy', arr=r_n)
    plt.plot(list(min_error_for_idx.keys()), list(min_error_for_idx.values()))
    plt.show()
    print(f'Max error = {max(min_error_for_idx.values())}, avg error: {np.mean(np.array(list(min_error_for_idx.values())))}')

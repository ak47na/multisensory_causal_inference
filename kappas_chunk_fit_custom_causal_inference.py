import multiprocessing as mp
import os
import numpy as np
from scipy.stats import vonmises, circmean
import utils
import plots
import matplotlib.pyplot as plt
import argparse
import pickle
from tqdm import tqdm

def compute_error(computed_values, data_slice):
    return utils.circular_dist(computed_values, data_slice)

def init_worker(angle_gam_data_path, unif_fn_data_path):
    global causal_inference_estimator
    global unif_map
    # Import JAX-related modules inside the worker initializer
    from custom_causal_inference import CustomCausalInference
    import forward_models_causal_inference

    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map
    np.random.seed(os.getpid())

def process_mean_pair(args):
    """
    Processes pairs of means and finds the optimal kappa values that minimize the error
    between model predictions and GAM data for a chunk of kappa combinations.

    Parameters:
        args (tuple): A tuple containing the following elements:
            - mean_indices (array-like): Indices of mean values to process.
            - ut (array-like): Array of transformed target means (ut).
            - us_n (array-like): Array of transformed sensory means (us_n).
            - kappa1_flat (array-like): Flattened array of kappa1 values.
            - kappa2_flat (array-like): Flattened array of kappa2 values.
            - num_sim (int): Number of simulations to run.
            - data_slice (array-like): GAM predicted response corresponding to the mean indices.
            - p_common (float): Probability of a common cause.
            - kappa_indices (array-like): Indices to select a chunk of kappa combinations.

    Returns:
        tuple: A tuple containing:
            - mean_indices (array-like): Indices of the processed mean values.
            - p_common (float): Probability of a common cause used.
            - optimal_kappas (tuple): A tuple of optimal kappa1 and kappa2 values minimizing the error.
            - min_error (array-like): Minimum error achieved with the optimal kappa values.
    """
    mean_indices, ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices = args

    mu1 = ut[mean_indices]
    mu2 = us_n[mean_indices]

    # Select the chunk of kappa combinations
    kappa1_chunk = kappa1_flat[kappa_indices]
    kappa2_chunk = kappa2_flat[kappa_indices]

    # Generate samples for running causal inference with concentrations from the kappa chunk
    t_samples, s_n_samples = causal_inference_estimator.get_vm_samples(
        num_sim=num_sim,
        mu_t=mu1,
        mu_s_n=mu2,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)

    # Find the (circular) mean of (causal inference) optimal responses across all (t, s_n) samples
    responses, posterior_p_common, mean_t_est, mean_sn_est = causal_inference_estimator.forward(
        t_samples=t_samples,
        s_n_samples=s_n_samples,
        p_common=p_common,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)
    del t_samples, s_n_samples

    errors = compute_error(mean_sn_est, data_slice)
    assert mean_sn_est.ndim == 2, f'Found mean_sn_est_shape={mean_sn_est.shape}'

    # Find the pair of kappas with minimum error between r_n(s_n, t) and the mean optimal estimate
    idx_min = np.argmin(errors, axis=1)
    optimal_kappa1 = kappa1_chunk[idx_min]
    optimal_kappa2 = kappa2_chunk[idx_min]
    min_error = np.min(errors, axis=1)
    print("Process ID: {}, mean shapes: {}, {}, min error {}".format(os.getpid(), mu1.shape, mu2.shape, min_error))

    return (mean_indices, p_common, (optimal_kappa1, optimal_kappa2), min_error)

def find_optimal_kappas():
    """
    Finds the optimal kappa values that minimize the error between model predictions and GAM data
    by processing chunks of kappa combinations in parallel using multiprocessing.

    This function prepares tasks for different combinations of mean indices, probabilities of a common cause (p_common),
    and chunks of kappa values. It then uses a multiprocessing pool to process these tasks concurrently.
    After processing, it collects and combines the results to find the kappa pairs that achieve the minimum error
    for each mean index and p_common value.

    Returns:
        tuple: A tuple containing:
            - optimal_kappa_pairs (dict): A dictionary mapping (mean index, p_common) tuples to optimal kappa pairs (kappa1, kappa2).
            - min_error_for_idx_pc (dict): A dictionary mapping (mean index, p_common) tuples to the minimum error achieved.
    """
    tasks = []
    print(f'Fitting for num_means={ut.shape}, data_shape={r_n.shape}')

    # Adjust based on memory availability
    chunk_size = 500

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
        for result in tqdm(pool.imap_unordered(process_mean_pair, tasks), total=len(tasks)):
            results.append(result)


    # Collect and combine results across chunks of concentrations
    optimal_kappa_pairs = {}
    min_error_for_idx_pc = {(idx, pc): np.pi for idx in range(r_n.shape[0]) for pc in p_commons}

    # Find the minimum error across kappa chunks
    for mean_indices, p_common, optimal_kappa_pair, min_error in results:
        idx = mean_indices[0]  # mean_indices is an array of one element *for now*
        key = (idx, p_common)
        if (key not in optimal_kappa_pairs) or (min_error < min_error_for_idx_pc[key]):
            optimal_kappa_pairs[key] = (optimal_kappa_pair[0], optimal_kappa_pair[1])
            min_error_for_idx_pc[key] = min_error

    return optimal_kappa_pairs, min_error_for_idx_pc

if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Fit kappas for grid pairs as specified by arguments.")
    parser.add_argument('--use_high_cc_error_pairs', type=bool, default=False, help='True if grid pairs are selected based on cue combination errors')

    num_sim = 1000
    D = 250  # grid dimension
    angle_gam_data_path = 'D:/AK_Q1_2024/Gatsby/data/base_bayesian_contour_1_circular_gam/base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path = 'D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl'
    p_commons = [0, .2, .5, .7, 1]
    args = parser.parse_args()
    use_high_cc_error_pairs = args.use_high_cc_error_pairs

    # Import JAX-related modules inside the main block
    from custom_causal_inference import CustomCausalInference
    from repulsion_hypothesis import repulsion_value
    import forward_models_causal_inference

    # Initialize the estimator inside the main block
    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map

    if use_high_cc_error_pairs:
        s_n, t, r_n = utils.get_cc_high_error_pairs(causal_inference_estimator.grid,
                                        causal_inference_estimator.gam_data,
                                        max_samples=1)
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

    min_kappa1, max_kappa1, num_kappa1s = 1, 200, 100
    min_kappa2, max_kappa2, num_kappa2s = 1.1, 300, 100
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

    optimal_kappa_pairs, min_error_for_idx_pc = find_optimal_kappas()
    print(f'Completed with optimal results = {optimal_kappa_pairs}')
    min_error_for_idx = {}
    for key in min_error_for_idx_pc:
        if key[0] in min_error_for_idx:
            min_error_for_idx[key[0]] = min(min_error_for_idx[key[0]], min_error_for_idx_pc[key])
        else:
            min_error_for_idx[key[0]] = min_error_for_idx_pc[key]

    plt.plot(list(min_error_for_idx.keys()), list(min_error_for_idx.values()))
    plt.show()

    # Save optimal parameters
    with open('./learned_data/optimal_kappa_pairs.pkl', 'wb') as f:
        pickle.dump(optimal_kappa_pairs, f)
    with open('./learned_data/min_error_for_idx_pc.pkl', 'wb') as f:
        pickle.dump(min_error_for_idx_pc, f)
    with open('./learned_data/min_error_for_idx.pkl', 'wb') as f:
        pickle.dump(min_error_for_idx, f)
    np.save('./learned_data/selected_s_n.npy', arr=s_n)
    np.save('./learned_data/selected_t.npy', arr=t)
    np.save('./learned_data/selected_r_n.npy', arr=r_n)
    print(f'Max error = {max(min_error_for_idx.values())}, avg error: {np.mean(np.array(list(min_error_for_idx.values())))}')
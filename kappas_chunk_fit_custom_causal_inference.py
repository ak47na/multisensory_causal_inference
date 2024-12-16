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
from custom_causal_inference import CustomCausalInference
import forward_models_causal_inference

def compute_error(computed_values, data_slice):
    return utils.circular_dist(computed_values, data_slice)

def init_worker(angle_gam_data_path, unif_fn_data_path):
    global causal_inference_estimator
    global unif_map

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
    task_idx, mean_indices, ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices = args
    max_to_save=50
    error_threshold=0.0349066 

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
    if max_to_save > 0:
        mean_min_indices,  kappas_min_indices = np.where(errors < error_threshold)
        if (len(mean_min_indices) > max_to_save) or (len(mean_min_indices) < 2):
            sorted_indices = np.argsort(errors, axis=None)
            # Convert flattened indices back to 2D (row, column) indices
            mean_min_indices,  kappas_min_indices = np.unravel_index(sorted_indices, errors.shape)
            mean_min_indices = mean_min_indices[:max_to_save]
            kappas_min_indices = kappas_min_indices[:max_to_save]
        
        errors_dict = {'errors': errors[mean_min_indices, kappas_min_indices],
                       'optimal_kappa1': np.round(kappa1_flat[kappas_min_indices], 4),
                       'optimal_kappa2': np.round(kappa2_flat[kappas_min_indices], 4)}
        
        with open (f'./learned_data/errors_dict_{task_idx}.pkl', 'wb') as f:
            pickle.dump(errors_dict, f)
    
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
    task_idx = 0
    task_metadata = {}
    for i, data_slice in enumerate(r_n):
        for p_common in p_commons:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_kappa_combinations)
                kappa_indices_chunk = kappa_indices[start_idx:end_idx]
                tasks.append((task_idx, np.array([i]), ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices_chunk))
                task_metadata[task_idx] = {'mean_indices': np.array([i]),
                                           'p_common': p_common,
                                           'kappa_indices': (start_idx, end_idx)}
                task_idx += 1
    with open('./learned_data/task_metadata.pkl', 'wb') as f:
        pickle.dump(task_metadata, f)
    initargs = (angle_gam_data_path, unif_fn_data_path)
    print("Before creating multiprocessing pool")
    #num_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
    num_processes = os.cpu_count()
    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=initargs) as pool:
        results = []
        print("Multiprocessing pool created")
        for result in pool.imap_unordered(process_mean_pair, tasks):
            results.append(result)
            print(f'Num results: {len(results)}, completed={100*len(results)/len(tasks)}%')
    print("After multiprocessing pool!")
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
    parser.add_argument('--use_unif_internal_space', type=int, default=0, help='If nonzero, number of s_n, t values to be selected as uniform values in internal space')
    num_sim = 1000
    D = 250  # grid dimension
    angle_gam_data_path = './base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path = './uniform_model_base_inv_kappa_free.pkl'
    p_commons = np.linspace(0, 1, num=10)
    args = parser.parse_args()
    use_high_cc_error_pairs = args.use_high_cc_error_pairs
    use_unif_internal_space = args.use_unif_internal_space

    # Initialize the estimator inside the main block
    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map

    if use_high_cc_error_pairs:
        assert (use_unif_internal_space ==0)
        s_n, t, r_n = utils.get_cc_high_error_pairs(causal_inference_estimator.grid,
                                        causal_inference_estimator.gam_data,
                                        max_samples=1)
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
    elif use_unif_internal_space != 0:
        assert (use_unif_internal_space > 0)
        # Select indices from quadrant [-np.pi, -np.pi/2)
        indices = 250//4+utils.select_evenly_spaced_integers(num=use_unif_internal_space, start=0, end=250//4)
        stimuli = np.linspace(-np.pi, np.pi, D)
        selected_internal_stimuli = stimuli[indices] # Uniform in internal space
        selected_stimuli = unif_map.unif_space_to_angle_space(selected_internal_stimuli)
        grid_indices_selected_stimuli = utils.select_closest_values(array=stimuli, 
                                                                    selected_values=selected_stimuli, 
                                                                    distance_function=utils.circular_dist)
        print(f'Indices in grid of selected stimuli: {grid_indices_selected_stimuli}')
        if (grid_indices_selected_stimuli[0] == 0) and (grid_indices_selected_stimuli[-1] == 0):
            grid_indices_selected_stimuli[-1] = D-1
        grid_indices_selected_stimuli = np.sort(grid_indices_selected_stimuli)
        print(f'Indices in grid of selected stimuli after wrap test: {grid_indices_selected_stimuli}')
        plt.scatter(selected_internal_stimuli, stimuli[grid_indices_selected_stimuli], label='selected s_n', alpha=.5, c='b')
        plt.scatter(selected_internal_stimuli, selected_stimuli, label='s_n uniform in internal space', alpha=.5, c='r')
        plt.scatter(selected_internal_stimuli, selected_internal_stimuli, alpha=.7, c='k', marker='x', label='s_n uniform in angle space')
        plt.legend()
        plt.show()
        grid_indices_selected_stimuli = np.unique(grid_indices_selected_stimuli)
        r_n = causal_inference_estimator.gam_data['full_pdf_mat'][grid_indices_selected_stimuli, :, 2]
        r_n = r_n[:, grid_indices_selected_stimuli]
        t, s_n = np.meshgrid(stimuli[grid_indices_selected_stimuli], 
                             stimuli[grid_indices_selected_stimuli], indexing='ij')
        plt.scatter(s_n, r_n, label='r_n as fn of s_n')
        plt.legend()
        plt.show()
        plt.scatter(np.arange(len(grid_indices_selected_stimuli)), grid_indices_selected_stimuli)
        plt.title('Indices of selected stimuli')
        plt.show()
        plt.scatter(grid_indices_selected_stimuli, grid_indices_selected_stimuli)
        plt.title('Indices of selected stimuli')
        plt.show()
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
        plots.heatmap_f_s_n_t(f_s_n_t=r_n, s_n=s_n, t=t, f_name='r_n')
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

    min_kappa1, max_kappa1, num_kappa1s = 1, 200, 10
    min_kappa2, max_kappa2, num_kappa2s = 1.1, 300, 10
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

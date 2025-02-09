import multiprocessing as mp
import os
import numpy as np
import utils
import plots
import matplotlib.pyplot as plt
import argparse
import pickle
import logging
from custom_causal_inference import CustomCausalInference
import forward_models_causal_inference
import submitit
import shutil
import sys
from submitit.helpers import as_completed

logger = logging.getLogger(__name__)

class KappaFitter:
    """
    A class to handle kappa fitting either locally using multiprocessing
    or on a SLURM cluster using Submitit.
    """

    def __init__(self,
                 ut,
                 us_n,
                 r_n,
                 kappa1_flat,
                 kappa2_flat,
                 p_commons,
                 num_sim,
                 angle_gam_data_path,
                 unif_fn_data_path,
                 local_run,
                 user):
        self.ut = ut
        self.us_n = us_n
        self.r_n = r_n
        self.kappa1_flat = kappa1_flat
        self.kappa2_flat = kappa2_flat
        self.p_commons = p_commons
        self.num_sim = num_sim
        self.angle_gam_data_path = angle_gam_data_path
        self.unif_fn_data_path = unif_fn_data_path
        self.local_run = local_run
        self.user = user

    def find_optimal_kappas(self):
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
        print(f'Fitting for num_means={self.ut.shape}, data_shape={self.r_n.shape}')
        print(f'User={self.user}')
        # Adjust based on memory availability
        if self.local_run:
            chunk_size = 500
        else:
            chunk_size = 10000

        total_kappa_combinations = len(self.kappa1_flat)
        kappa_indices = np.arange(total_kappa_combinations)

        num_chunks = (total_kappa_combinations + chunk_size - 1) // chunk_size
        task_idx = 0
        task_metadata = {}
        for i, data_slice in enumerate(self.r_n):
            for p_common in self.p_commons:
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, total_kappa_combinations)
                    kappa_indices_chunk = kappa_indices[start_idx:end_idx]
                    tasks.append((task_idx,
                                  np.array([i]),
                                  self.ut,
                                  self.us_n,
                                  self.kappa1_flat,
                                  self.kappa2_flat,
                                  self.num_sim,
                                  data_slice,
                                  p_common,
                                  kappa_indices_chunk))
                    task_metadata[task_idx] = {
                        'mean_indices': np.array([i]),
                        'p_common': p_common,
                        'kappa_indices': (start_idx, end_idx)
                    }
                    task_idx += 1

        with open('./learned_data/task_metadata.pkl', 'wb') as f:
            pickle.dump(task_metadata, f)

        if self.local_run:
            initargs = (self.angle_gam_data_path, self.unif_fn_data_path)
            logger.debug("Before creating multiprocessing pool")
            num_processes = os.cpu_count()
            with mp.Pool(processes=num_processes,
                         initializer=init_worker,
                         initargs=initargs) as pool:
                results = []
                logger.debug(f"Multiprocessing pool created for {len(tasks)} tasks")
                for result in pool.imap_unordered(process_mean_pair, tasks):
                    results.append(result)
                    logger.debug(f'Num results: {len(results)}, '
                                 f'completed={100*len(results)/len(tasks):.2f}%')
            logger.debug("After multiprocessing pool!")
            # Collect and combine results across chunks of concentrations
            optimal_kappa_pairs, min_error_for_idx_pc = report_min_error(
                results, self.p_commons, self.r_n.shape[0])
            return optimal_kappa_pairs, min_error_for_idx_pc
        else:
            log_folder = f'/ceph/scratch/{self.user}/slurm/logs/%j'
            print(f'Running on the cluser, {len(tasks)} tasks')
            # Create tmp directory for logging (logs will be deleted after the job terminates)
            try:
                os.makedirs(log_folder, exist_ok=False)
                logger.critical(f"Directory '{log_folder}' created successfully.")
            except Exception as e:
                logger.critical(f"Error creating directory '{log_folder}': {e}")
                exit(1)

            executor = submitit.AutoExecutor(folder=log_folder)
            num_processes = 8
            # slurm_array_parallelism tells the scheduler to only run at most 16 jobs at once.
            # By default, this is several hundreds (no HPC default!)
            executor.update_parameters(slurm_array_parallelism=16,
                                       slurm_partition='cpu',
                                       timeout_min=1000,
                                       mem_gb=32,
                                       cpus_per_task=num_processes)
            jobs = executor.map_array(process_mean_pair, tasks)
            logger.debug('Before running results')
            job_ids = [job.job_id for job in jobs]
            results = []
            for job in as_completed(jobs):
                try:
                    result = job.result()  # Blocks until this specific job finishes
                    logger.debug(f"Job {job.job_id} completed: {len(results) + 1}/{len(jobs)}")
                    results.append(result)

                    # Delete the job’s log folder
                    job_folder = job.paths.folder
                    shutil.rmtree(job_folder)
                    logger.debug(f"Deleted log folder for job {job.job_id}: {job_folder}")
                except Exception as e:
                    logger.debug(f"Job {job.job_id} failed: {e}")
                    try:
                        job_folder = job.paths.folder
                        shutil.rmtree(job_folder)
                        logger.debug(f"Deleted log folder for job {job.job_id}: {job_folder}")
                    except Exception as e2:
                        logger.debug(f"Error deleting log folder for job {job.job_id}: {job_folder}: {e2}")

            # Collect and combine results across chunks of concentrations
            logger.debug('Combining results ...')
            report_min_executor = submitit.AutoExecutor(folder=log_folder)
            report_min_executor.update_parameters(
                slurm_partition='cpu',
                timeout_min=1000,
                mem_gb=32,
                cpus_per_task=num_processes,
                # Set up Slurm dependency so that this job starts
                # only after ALL listed job IDs complete successfully
                slurm_additional_parameters={
                    "dependency": "afterok:" + ":".join(job_ids)
                }
            )
            logger.debug('Before running min job')
            min_job = report_min_executor.submit(report_min_error,
                                                 results,
                                                 self.p_commons,
                                                 self.r_n.shape[0])
            optimal_kappa_pairs, min_error_for_idx_pc = min_job.result()
            # Delete the log folder
            try:
                shutil.rmtree(log_folder[:-3])  # Remove the '/%j' suffix
                logger.debug(f"Log directory '{log_folder}' has been deleted.")
            except Exception as e:
                logger.debug(f"Error deleting log directory '{log_folder}': {e}")
            return optimal_kappa_pairs, min_error_for_idx_pc


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

def process_mean_pair(args):
    """
    Processes pairs of means and finds the optimal kappa values that minimize the error
    between model predictions and GAM data for a chunk of kappa combinations.
    ...
    """
    task_idx, mean_indices, ut, us_n, kappa1_flat, kappa2_flat, num_sim, data_slice, p_common, kappa_indices = args
    max_to_save=50
    error_threshold=0.0349066

    mu1 = ut[mean_indices]
    mu2 = us_n[mean_indices]
    np.random.seed(os.getpid())

    # Select the chunk of kappa combinations
    kappa1_chunk = kappa1_flat[kappa_indices]
    kappa2_chunk = kappa2_flat[kappa_indices]

    # Generate samples for running causal inference with concentrations from the kappa chunk
    # t_samples, s_n_samples shape: [len(mean_indices), len(kappa1_flat), num_sim]
    t_samples, s_n_samples = causal_inference_estimator.get_vm_samples(
        num_sim=num_sim,
        mu_t=mu1,
        mu_s_n=mu2,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)
    
    # Find the (circular) mean of (causal inference) optimal responses across all (t, s_n) samples
    _, _, _, mean_sn_est = causal_inference_estimator.forward(
        t_samples=t_samples,
        s_n_samples=s_n_samples,
        p_common=p_common,
        kappa1=kappa1_chunk,
        kappa2=kappa2_chunk)
    del t_samples, s_n_samples

    errors = compute_error(mean_sn_est, data_slice) # shape is [len(mean_indices), len(kappa1_flat)]
    assert mean_sn_est.ndim == 2, f'Found mean_sn_est_shape={mean_sn_est.shape}'
    del mean_sn_est

    idx_min = np.argmin(errors, axis=1)
    optimal_kappa1 = kappa1_chunk[idx_min]
    optimal_kappa2 = kappa2_chunk[idx_min]
    min_error = np.min(errors, axis=1)
    if max_to_save > 0:
        mean_min_indices,  kappas_min_indices = np.where(errors < error_threshold)
        if (len(mean_min_indices) > max_to_save) or (len(mean_min_indices) < 2):
            # Edge cases: too many or too few "good kappas" with small error values to be saved.
            sorted_indices = np.argsort(errors, axis=None)
            # Indices in sorted_indices corresponed to a flattened errors array
            # Convert 1D indices of flattened errors back to 2D (row, column) indices
            mean_min_indices,  kappas_min_indices = np.unravel_index(sorted_indices, errors.shape)
            # Select only the first max_to_save indices (sorting ensures we select the best values)
            mean_min_indices = mean_min_indices[:max_to_save]
            kappas_min_indices = kappas_min_indices[:max_to_save]
        
        # Save the lowest errors and associated concentraions/kappa pairs
        # Grid indices of mean stimuli values (and p_common) are identified using the task_idx data
        # in task_metadata
        errors_dict = {'errors': errors[mean_min_indices, kappas_min_indices],
                       'optimal_kappa1': np.round(kappa1_flat[kappas_min_indices], 4),
                       'optimal_kappa2': np.round(kappa2_flat[kappas_min_indices], 4)}
        
        with open (f'./learned_data/optimal_kappa_errors/errors_dict_{task_idx}.pkl', 'wb') as f:
            pickle.dump(errors_dict, f)
        del errors_dict
        # Call gc.collect() if experiencing memory issues

    return (mean_indices, p_common, (optimal_kappa1, optimal_kappa2), min_error)


def report_min_error(results, p_commons, num_data_points):
    optimal_kappa_pairs = {}
    min_error_for_idx_pc = {(idx, pc): np.pi for idx in range(num_data_points) for pc in p_commons}

    # Find the minimum error across kappa chunks
    for mean_indices, p_common, optimal_kappa_pair, min_error in results:
        idx = mean_indices[0] # mean_indices is an array of one element *for now*
        key = (idx, p_common)
        if (key not in optimal_kappa_pairs) or (min_error[0] < min_error_for_idx_pc[key]):
            optimal_kappa_pairs[key] = (optimal_kappa_pair[0], optimal_kappa_pair[1])
            min_error_for_idx_pc[key] = min_error[0] # min_error has the same shape as mean_indices
    return optimal_kappa_pairs, min_error_for_idx_pc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit kappas for grid pairs as specified by arguments.")
    parser.add_argument('--debug', action='store_true',
                        help="If set, prints/log statements are enabled at DEBUG level.")
    parser.add_argument('--num_kappa1s', type=int, default=100,
                        help="Number of kappa values to be used for fitting kappa1")
    parser.add_argument('--num_kappa2s', type=int, default=100,
                        help="Number of kappa values to be used for fitting kappa2")
    parser.add_argument('--num_sim', type=int, default=1000,
                        help="Number of simulations to be run for each kappa pair")
    parser.add_argument('--t_index', type=int, default=2,
                        help="Index of the regressor t used for regressed r_n(s_n, t)")
    parser.add_argument('--user', type=str, default='',
                        help="Username for running user, used for selecting the log folder")
    parser.add_argument('--local_run', type=bool, default=False,
                        help='True if the script runs locally using multiprocessing')
    parser.add_argument('--use_high_cc_error_pairs', type=bool, default=False,
                        help='True if grid pairs are selected based on cue combination errors')
    parser.add_argument('--use_unif_internal_space', type=int, default=0,
                        help='If nonzero, number of s_n, t values to be selected as uniform values in internal space')
    D = 250
    p_commons = np.linspace(0, 1, num=20)

    args = parser.parse_args()

    # Configure logging: If --debug is set, log at DEBUG level; otherwise CRITICAL (effectively no output).
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.CRITICAL, format='[%(levelname)s] %(message)s')

    num_sim = args.num_sim
    t_index = args.t_index
    user = args.user
    local_run = args.local_run
    use_high_cc_error_pairs = args.use_high_cc_error_pairs
    use_unif_internal_space = args.use_unif_internal_space
    data_pref = '.'
    if not local_run:
        data_pref = '/nfs/ghome/live/kdusterwald/Documents/causal_inf'
    angle_gam_data_path = f'{data_pref}/base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path = f'{data_pref}/uniform_model_base_inv_kappa_free.pkl'

    causal_inference_estimator = forward_models_causal_inference.CausalEstimator(
        model=CustomCausalInference(decision_rule='mean'),
        angle_gam_data_path=angle_gam_data_path,
        unif_fn_data_path=unif_fn_data_path)
    unif_map = causal_inference_estimator.unif_map

    if use_high_cc_error_pairs:
        assert (use_unif_internal_space == 0)
        s_n, t, r_n = utils.get_cc_high_error_pairs(causal_inference_estimator.grid,
                                                    causal_inference_estimator.gam_data,
                                                    max_samples=1)
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
    elif use_unif_internal_space != 0:
        assert (use_unif_internal_space > 0)
        # Select indices from quadrant [-np.pi, -np.pi/2)
        indices = 250 // 4 + utils.select_evenly_spaced_integers(num=use_unif_internal_space,
                                                                start=0,
                                                                end=250 // 4)
        stimuli = np.linspace(-np.pi, np.pi, D)
        selected_internal_stimuli = stimuli[indices] # Uniform stimuli in internal space
        # Convert to angles because data is in angle space
        selected_stimuli = unif_map.unif_space_to_angle_space(selected_internal_stimuli)
        # Bin the angles to the 250 discrete angle values in our dataset
        grid_indices_selected_stimuli = utils.select_closest_values(
            array=stimuli,
            selected_values=selected_stimuli,
            distance_function=utils.circular_dist)
        print(f'Indices in grid of selected stimuli: {grid_indices_selected_stimuli}')
        # Handle the wrap
        if (grid_indices_selected_stimuli[0] == 0) and (grid_indices_selected_stimuli[-1] == 0):
            grid_indices_selected_stimuli[-1] = D - 1
        grid_indices_selected_stimuli = np.sort(grid_indices_selected_stimuli)
        print(f'Indices in grid of selected stimuli after wrap test: {grid_indices_selected_stimuli}')
        if local_run:
            plt.scatter(selected_internal_stimuli, stimuli[grid_indices_selected_stimuli],
                        label='selected s_n', alpha=.5, c='b')
            plt.scatter(selected_internal_stimuli, selected_stimuli,
                        label='s_n uniform in internal space', alpha=.5, c='r')
            plt.scatter(selected_internal_stimuli, selected_internal_stimuli,
                        alpha=.7, c='k', marker='x', label='s_n uniform in angle space')
            plt.legend()
            plt.show()
        grid_indices_selected_stimuli = np.unique(grid_indices_selected_stimuli)
        r_n = causal_inference_estimator.gam_data['full_pdf_mat'][grid_indices_selected_stimuli, :, t_index]
        r_n = r_n[:, grid_indices_selected_stimuli]
        t, s_n = np.meshgrid(stimuli[grid_indices_selected_stimuli],
                             stimuli[grid_indices_selected_stimuli],
                             indexing='ij')
        if local_run:
            plt.scatter(s_n, r_n, label='r_n as fn of s_n')
            plt.legend()
            plt.show()
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
        plots.heatmap_f_s_n_t(f_s_n_t=r_n, s_n=s_n, t=t, f_name='r_n')
    else:
        s_n, t, r_n = utils.get_s_n_and_t(causal_inference_estimator.grid,
                                          causal_inference_estimator.gam_data)
        print(f'Shapes of s_n, t, and r_n means: {s_n.shape, t.shape, r_n.shape}')
        # Further filtering
        num_means = 4
        step = len(s_n) // num_means
        indices = np.arange(0, s_n.shape[0], step=step)
        s_n = s_n[indices][:, indices]
        t = t[indices][:, indices]
        r_n = r_n[indices][:, indices]
        plots.heatmap_f_s_n_t(f_s_n_t=r_n, s_n=s_n, t=t, f_name='r_n')

    min_kappa1, max_kappa1, num_kappa1s = 1, 200, args.num_kappa1s
    min_kappa2, max_kappa2, num_kappa2s = 1.1, 300, args.num_kappa2s
    s_n, t, r_n = s_n.flatten(), t.flatten(), r_n.flatten()
    us_n = unif_map.angle_space_to_unif_space(s_n)
    ut = unif_map.angle_space_to_unif_space(t)
    kappa1 = np.logspace(start=np.log10(min_kappa1),
                         stop=np.log10(max_kappa1),
                         num=num_kappa1s,
                         base=10)
    kappa2 = np.logspace(start=np.log10(min_kappa2),
                         stop=np.log10(max_kappa2),
                         num=num_kappa2s,
                         base=10)
    kappa1_grid, kappa2_grid = np.meshgrid(kappa1, kappa2, indexing='ij')
    kappa1_flat, kappa2_flat = kappa1_grid.flatten(), kappa2_grid.flatten()
    print(f'Performing causal inference for ut, us_n of shape {ut.shape, us_n.shape}')
    if local_run:
        plt.plot(kappa1, label='kappa1')
        plt.plot(kappa2, label='kappa2')
        plt.legend()
        plt.show()

    fitter = KappaFitter(ut=ut,
                         us_n=us_n,
                         r_n=r_n,
                         kappa1_flat=kappa1_flat,
                         kappa2_flat=kappa2_flat,
                         p_commons=p_commons,
                         num_sim=num_sim,
                         angle_gam_data_path=angle_gam_data_path,
                         unif_fn_data_path=unif_fn_data_path,
                         local_run=local_run,
                         user=user)

    optimal_kappa_pairs, min_error_for_idx_pc = fitter.find_optimal_kappas()
    print(f'Completed with optimal results = {optimal_kappa_pairs}')
    min_error_for_idx = {}
    for key in min_error_for_idx_pc:
        if key[0] in min_error_for_idx:
            min_error_for_idx[key[0]] = min(min_error_for_idx[key[0]], min_error_for_idx_pc[key])
        else:
            min_error_for_idx[key[0]] = min_error_for_idx_pc[key]
    if local_run:
        plt.plot(list(min_error_for_idx.keys()), list(min_error_for_idx.values()))
        plt.show()
    with open('./learned_data/optimal_kappa_pairs_4.pkl', 'wb') as f:
        pickle.dump(optimal_kappa_pairs, f)
    with open('./learned_data/min_error_for_idx_pc_4.pkl', 'wb') as f:
        pickle.dump(min_error_for_idx_pc, f)
    with open('./learned_data/min_error_for_idx_4.pkl', 'wb') as f:
        pickle.dump(min_error_for_idx, f)
    np.save('./learned_data/selected_s_n.npy', arr=s_n)
    np.save('./learned_data/selected_t.npy', arr=t)
    np.save('./learned_data/selected_r_n.npy', arr=r_n)
    print(f'Max error = {max(min_error_for_idx.values())}, '
                 f'avg error: {np.mean(list(min_error_for_idx.values()))}')

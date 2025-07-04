import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import mode


def reflect_cond(t, s_n):
    if (np.sign(t) == np.sign(s_n)) or (abs(s_n) < 1e-6):
        return False
    return True

def get_filtered_gam(gam_data, max_circ_dist, filter_wrong_quad=False, r_n_key='r_n'):
    filtered_gam_data = {'reg_dict': {key: [] for key in gam_data['reg_dict'].keys()}}
    for i in range(len(gam_data['reg_dict']['s_n'])):
        s_n, r_n = gam_data['reg_dict']['s_n'][i], gam_data['reg_dict'][r_n_key][i]
        if circular_dist(s_n, r_n) < max_circ_dist:
            if filter_wrong_quad:
                # TODO: should we use the 4 quadrants or rather acc for the doubling of angles?
                if np.sign(s_n) != np.sign(r_n):
                    continue
            for key in filtered_gam_data['reg_dict'].keys():
                filtered_gam_data['reg_dict'][key].append(gam_data['reg_dict'][key][i])
    for key in filtered_gam_data['reg_dict'].keys():
        filtered_gam_data['reg_dict'][key] = np.array(filtered_gam_data['reg_dict'][key])
    return filtered_gam_data


def get_cc_high_error_pairs(grid, gam_data, max_samples=400, t_index=2):
    '''
    Returns at most max_samples random tuples of (s_n, t, r_n) from the grid pairs where no optimal
    solution could be found with cue combination.
    '''
    sol_mat = np.load('./learned_data/cue_comb_sol_for_grid.npy')
    assert sol_mat.shape[0] == grid.shape[0]
    t_indices, s_n_indices = np.where(sol_mat == 0)
    t, s_n = grid[t_indices], grid[s_n_indices]
    r_n = gam_data['full_pdf_mat'][t_indices, s_n_indices, t_index]
    max_samples = min(max_samples, len(t))
    indices = np.sort(np.random.choice(a=len(t), size=max_samples, replace=False))
    return s_n[indices], t[indices], r_n[indices]

def mu_kappa_shape_match(mu, kappa):
    if isinstance(mu, (int, float)) or isinstance(kappa, (int, float)):
        return True
    if mu.ndim == kappa.ndim:
        return (mu.shape == kappa.shape) or (mu.shape[:-1] == kappa.shape[:-1] and (mu.shape[-1] == 1))
    return (mu.shape[1:] == kappa.shape)

def mus_shape_match(mu1, mu2):
    if isinstance(mu1, (int, float)) or isinstance(mu2, (int, float)):
        return True
    return (mu1.shape == mu2.shape)


def get_s_n_and_t(grid, gam_data, step=1, t_index=2):
    indices = np.arange(len(grid), step=step)
    t, s_n = np.meshgrid(grid[indices], grid[indices], indexing='ij')
    r_n = gam_data['full_pdf_mat'][indices, :, t_index]
    r_n = r_n[:, indices]
    return s_n, t, r_n


def estimate_mu_and_kappa_von_mises(angles, axis=0):
    """
    Estimate the mean and concentration parameter kappa of a Von Mises distribution along the axis
    axis of the ndarray `angles`.

    Parameters:
    angles (ndarray): Input array of angles (in radians), where the axis axis corresponds to the
                      number of samples, and the remaining dimensions correspond to different
                      distributions.

    Returns:
    mu(ndarray): Estimated mean parameter along the axis dimension.
    kappa (ndarray): Estimated concentration parameter along the axis dimension.
    """

    # Compute the cosine and sine of the angles
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    # Sum along the axis axis
    sum_cos = np.sum(cos_angles, axis=axis)
    sum_sin = np.sum(sin_angles, axis=axis)
    # Estimate the mean
    mu = np.arctan2(sum_sin, sum_cos)
    # Number of samples
    N = angles.shape[0]
    # Compute the resultant vector length R
    R = np.sqrt(sum_cos**2 + sum_sin**2) / N
    # Estimate kappa using the approximation
    # Handling the small R case separately if needed
    kappa = np.where(
        R < 0.53,
        2 * R + R**3 + (5 * R**5) / 6,
        np.where(
            R < 0.85,
            -0.4 + 1.39 * R + 0.43 / (1 - R),
            1 / (2 * (1 - R))
        )
    )
    return mu, kappa


def wrap(x: np.ndarray) -> np.ndarray:
    """
    Wrap the input angle(s) into the range [-pi, pi].

    Parameters:
        x : np.ndarray
            The input angle(s) to be wrapped, represented as an array of angles.

    Returns:
        np.ndarray:
            The wrapped angle(s) with values in the range [-pi, pi].

    Notes:
        The function wraps the input angle(s) into the range [-pi, pi] by adding 3*pi, taking the modulo by 2*pi,
        and then subtracting pi.

        This function is useful for normalizing angles to a consistent range.
    """
    return np.mod(x + 3 * np.pi, 2 * np.pi) - np.pi


def abs_dist(x, y):
    return np.abs(x-y)

def circular_dist(x, y):
    """
    Calculate the circular distance between two angles x and y.
    Both x and y should be in radians.
    """
    return np.pi - np.abs((np.mod(np.abs(x - y), 2*np.pi)) - np.pi)


def plot_2d_histogram(x_1s, x_2s, n_bins=50):
    # Flatten the x_1s and x_2s arrays
    x_1_flat = x_1s.flatten()
    x_2_flat = x_2s.flatten()
    print(f'x_1_min_max={np.min(x_1_flat), np.max(x_1_flat)}')
    print(f'x_2_min_max={np.min(x_2_flat), np.max(x_2_flat)}')
    # Plot the 2D histogram
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_1_flat, x_2_flat, bins=n_bins, density=True, cmap='Blues')
    plt.colorbar(label='Probability Density')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Histogram of $p(x_1, x_2)$')
    plt.grid(True)
    plt.show()

def hist_bins(num: int = 51) -> Tuple[np.ndarray, float]:
    """
    Create bin edges and bin width for a circular histogram.

    Parameters:
        num : int, optional (default=51)
            The number of bins in the histogram.

    Returns:
        Tuple[np.ndarray, float]:
            An array representing the bin edges and the bin width.

    Notes:
        The function generates bin edges and bin width for a circular histogram that spans the range [-pi, pi].

        The bin edges are linearly spaced points that span the range from -pi to pi, divided into num intervals.
        The bin width is calculated as 2*pi/num.

        The function is useful for creating bins for circular data when constructing a histogram.
    """
    bin_edges = np.linspace(-np.pi, np.pi, num + 1, endpoint=True)
    bin_width = 2 * np.pi / num
    return bin_edges, bin_width


def modes(data: np.ndarray, num_bins: int) -> np.ndarray:
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    digitized = np.digitize(data, bins) - 1  # Subtract 1 to make bin indices start from 0
    
    # Compute the mode across the first axis (i.e., across num_points)
    modes_indices, _ = mode(digitized, axis=0, keepdims=False)
    # Map the bin indices back to the corresponding bin centers (optional)
    # Compute the bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Convert mode bin indices back to bin centers
    return bin_centers[modes_indices]


def select_evenly_spaced_integers(num, start=0, end=249):
    """
    Selects num as evenly spaced as possible integers in the range from start to end inclusive.

    Parameters:
    - num (int): Number of integers to select.
    - start (int): Start of the range (inclusive).
    - end (int): End of the range (inclusive).

    Returns:
    - indices (np.ndarray): Array of selected integers.
    """
    # Total number of available integers in the range
    total_integers = end - start + 1
    # Adjust num if it exceeds the total number of available integers
    num = min(num, total_integers)
    # Generate x evenly spaced values in the range
    indices = np.linspace(start, end, num)
    # Round to the nearest integer
    indices = np.round(indices).astype(int)
    # Ensure indices are within the bounds
    indices = np.clip(indices, start, end)
    # Remove duplicates (if any)
    indices = np.unique(indices)
    return indices


def select_closest_values(array, selected_values, distance_function):
    """
    Selects num_to_select elements from 'array' that are closest to any value in 'selected_values'.

    Parameters:
    - array (np.ndarray): 1D array from which to select values.
    - selected_values (np.ndarray): 1D array of values to compare against.
    - num_to_select (int): Number of closest values to select from 'array'.
    - distance_function (callable): Function to compute distance between elements.

    Returns:
    - selected_indices (np.ndarray): Indices of the selected values in 'array'.
    """
    # Compute the distance matrix between 'array' and 'selected_values'
    distances = distance_function(selected_values[:, np.newaxis], array[np.newaxis, :])
    # distances[i, j] = distance_function(selected_values[i], array[j])
    # Get the indices of the elements in array closest to elements in selected_values
    selected_indices = np.argmin(distances, axis=1)
    assert selected_indices.shape == selected_values.shape
    return selected_indices

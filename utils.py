import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import mode


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

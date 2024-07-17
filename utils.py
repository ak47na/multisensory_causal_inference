import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


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

def __bin_1d(data: np.ndarray, cur_mid: float, width: float) -> np.ndarray:
    """
    Internal function to perform one-dimensional binning.

    Parameters:
        data : jnp.ndarray
            The one-dimensional data to be binned.
        cur_mid : float
            The midpoint of the bin.
        width : float
            The width of the bin.

    Returns:
        jnp.ndarray:
            The boolean array representing whether each data point falls into the bin.

    Notes:
        This function performs binning of the one-dimensional data into a single bin defined by the current midpoint and width.

        It calculates the absolute difference between each data point and the current midpoint and compares it with the width/2
        to determine whether the data point falls into the bin.

        This function is used internally by bin_1d to perform the actual binning operation.
    """
    mod_data = np.abs(wrap(data - cur_mid))
    bin_data = np.where(mod_data < width/2,True,False)
    return bin_data
# _bin_1d = jit(vmap(__bin_1d,in_axes=(None,0,None)))

def bin_1d(data: np.ndarray, mids: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin one-dimensional data into discrete intervals.

    Parameters:
        data : np.ndarray
            The one-dimensional data to be binned.
        mids : np.ndarray
            The midpoints of the bins represented as a one-dimensional array.
        width : float
            The width of each bin.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Two arrays representing the counts and the binned data.

    Notes:
        The function bins the one-dimensional data into discrete intervals defined by the midpoints and width.

        The function uses _bin_1d internally to perform the actual binning.

        The counts array represents the number of data points falling into each bin, and the bin_data array
        represents the boolean values indicating whether each data point falls into the corresponding bin.

        This function is useful for histogramming and analyzing one-dimensional data.
    """
    bin_data = __bin_1d(data, mids, width)
    counts = np.sum(bin_data, axis=1)
    return counts, bin_data

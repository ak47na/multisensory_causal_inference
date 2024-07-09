import numpy as np
import matplotlib.pyplot as plt


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

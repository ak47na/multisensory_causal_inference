import numpy as np


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
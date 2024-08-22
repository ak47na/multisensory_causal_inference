# TODO(ak47na): if needed, refactor into an Interpolation interface and subclasses for different
# interpolation methods
import numpy as np
from utils import circular_dist, abs_dist, wrap
from bisect import bisect_left

def bs_vectorized(x, xs, distance_fn=abs_dist):
    if isinstance(x, float) or isinstance(x, int):
        return bs(x, xs)
    # Convert xs to a NumPy array for efficient processing
    xs = np.asarray(xs)

    # Use searchsorted to find the insertion points for all x values
    idx = np.searchsorted(xs, x, side='left')

    # Handle edge cases
    idx[idx == 0] = 1  # If x is beyond the left end, set idx to 1
    idx[idx == len(xs)] = len(xs) - 1  # If x is beyond the right end, set idx to len(xs)-1

    # Compute distances to the left and right neighbors
    left_dist = distance_fn(x, xs[idx - 1])
    right_dist = distance_fn(x, xs[idx])

    # Determine whether the left or right neighbor is closer
    closest_idx = np.where(left_dist <= right_dist, idx - 1, idx)

    return closest_idx

def bs(x, xs):
    # Find the insertion point
    idx = bisect_left(xs, x)
    # If x is beyond the left end
    if idx == 0:
        return 0
    # If x is beyond the right end
    elif idx == len(xs):
        return idx - 1
    # Otherwise, find the closest of the two neighbors
    else:
        if abs(x - xs[idx-1]) <= abs(x - xs[idx]):
            return idx - 1
        else:
            return idx

def find_closest_kappa(kappa, kappas):
    """
    Find the index of the closest kappa_i in kappas to the given kappa using binary search.
    kappas is assumed to be sorted.
    """
    return bs_vectorized(kappa, kappas)

def find_closest_mu(mu, mus):
    """
    Find the index of the closest mu_i in mus to the given mu using circular distance.
    """
    mu_len = 1
    mu = wrap(mu)
    if isinstance(mu, np.ndarray):
        mu_len = mu.shape
    else:
        assert (isinstance(mu, float)), f'Mu has type {type(mu)}, but only float and np.ndarray are supported'
    circular_distances = np.array([circular_dist(m, mu) for m in mus]).reshape(-1, *mu_len)
    #print(f'dist_shape={circular_distances.shape}')
    if isinstance(mu, float):
        return np.argmin(circular_dist, axis=0)[0]
    return np.argmin(circular_distances, axis=0)

def find_closest_mu_bs(mu, mus):
    """
    Find the index of the closest mu_i in mus to the given mu using binary search.
    mus is assumed to be sorted.
    """
    idx_1 = bs_vectorized(mu, mus, distance_fn=circular_dist)
    idx_2 = np.where(mu < 0, bs_vectorized(2*np.pi+mu, mus, distance_fn=circular_dist),
                     bs_vectorized(mu-2*np.pi, mus, distance_fn=circular_dist))
    circ_dist_to_idx1 = circular_dist(mu, mus[idx_1])
    circ_dist_to_idx2 = circular_dist(mu, mus[idx_2])
    return np.where(circ_dist_to_idx1 <= circ_dist_to_idx2, idx_1, idx_2)

def f_mu_kappa_scipy_interp(mu, kappa, interp):
    """
    Interpolates the value of f for given mu and kappa.

    Parameters:
    - mu: The mu value to interpolate at.
    - mu: The kappa value to interpolate at.
    - interpolator: The interpolator object created by RectBivariateSpline.

    Returns:
    - Interpolated value of f at (mu, mu).
    """
    return interp(mu, kappa)[0, 0]

def f_mu_kappa_grid_interp(mu, kappa, interp, f, return_idx=False, use_binary_search=False):
    assert isinstance(interp, dict)
    kappa_idx = find_closest_kappa(kappa=kappa, kappas=interp['kappas'])
    if use_binary_search:
        mu_idx = find_closest_mu_bs(mu=mu, mus=interp['mus'])
    else:
        mu_idx = find_closest_mu(mu, interp['mus'])
    if return_idx:
        return interp[f][mu_idx, kappa_idx], mu_idx, kappa_idx
    return interp[f][mu_idx, kappa_idx]

def f_mu_kappa(mu, kappa, interp, f, interpolation_type='grid'):
    assert interpolation_type in ['grid', 'scipy']
    assert (f in [None, 'r', 'R', 'r_grid'])
    if interpolation_type == 'grid':
        assert (f is not None)
        return f_mu_kappa_grid_interp(mu=mu, kappa=kappa, interp=interp, f=f)
    else:
        assert (f is None)
        return f_mu_kappa_scipy_interp(mu, kappa, interp)
    
def add_key_from_file_to_interp(interp, key, value_path):
    interp[key] =np.load(value_path)

def save_key_to_file(interp, key, value_path):
    np.save(interp[key], value_path)


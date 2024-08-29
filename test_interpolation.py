import numpy as np
import interpolation

def fixed_kappa_test(mu: np.ndarray, kappa: int, interp, expected_mu_idx, expected_kappa_idx, excepted_grid_vals, use_binary_search):
    grid_vals, mu_idx, kappa_idx = interpolation.f_mu_kappa_grid_interp(mu=mu, kappa=kappa, 
                                                                        interp=interp, f='R_mean',
                                                                        return_idx=True,
                                                                        use_binary_search=use_binary_search)
    assert (mu.shape == grid_vals.shape)
    assert np.array_equal(mu_idx, expected_mu_idx), f'{mu_idx, expected_mu_idx}'
    assert np.array_equal(kappa_idx, expected_kappa_idx)
    assert np.array_equal(grid_vals, excepted_grid_vals)
    print(f'Simple test for interpolating f(mu, kappa) passed!')


def test_find_closest_mu():
    mus = np.array([-np.pi, -np.pi/3, 0, np.pi/2, np.pi-.001])
    
    # Normal case
    mu = np.array([-np.pi/2, 0, np.pi/3])
    expected_idxs = np.array([1, 2, 3]) 
    assert np.array_equal(interpolation.find_closest_mu(mu, mus), expected_idxs)
    assert np.array_equal(interpolation.find_closest_mu_bs(mu, mus), expected_idxs)
    
    # Edge case: smaller than minimum and greater than maximum
    mu = np.array([-np.pi-0.3, np.pi+0.1])
    expected_idxs = np.array([4, 0])
    assert np.array_equal(interpolation.find_closest_mu(mu, mus), expected_idxs)
    assert np.array_equal(interpolation.find_closest_mu_bs(mu, mus), expected_idxs)
    
    print("test_find_closest_mu passed!")

def larger_test_find_closest_mu(num_mus=1000, num_mu=72):
    mus = np.linspace(-np.pi+.05, np.pi-.04, num=num_mus)
    mu = np.linspace(-np.pi, np.pi, num=num_mu)
    
    assert np.array_equal(interpolation.find_closest_mu(mu, mus),
                          interpolation.find_closest_mu_bs(mu, mus))
    print("larger_test_find_closest_mu passed!")

def test_bs_vectorized():
    xs = np.array([0.1, 0.5, 1.2, 1.8, 2.5, 3.3, 4.0])
    x_values = np.array([-0.5, 1.5, 0.4, 3.0, 5])  # Normal case
    expected_idxs = np.array([0, 2, 1, 5, 6])
    assert np.array_equal(interpolation.bs_vectorized(x_values, xs), expected_idxs)
    print("test_bs_vectorized passed!")

def test_find_closest_kappa():
    kappas = np.linspace(0.1, 10, num=10)
    
    # Normal case
    kappa = np.array([0.05, 12.0, 1.2, 4.5, 7.8])
    expected_idxs = np.array([0, 9, 1, 4, 7])
    assert np.array_equal(interpolation.find_closest_kappa(kappa, kappas), expected_idxs)
    print("test_find_closest_kappa passed!")


def test_f_mu_kappa_grid_interp(use_binary_search):
    interp = {
        'mus': np.array([-np.pi, -np.pi/2, 0, np.pi/2-0.01, np.pi-0.01]),
        'kappas': np.linspace(0.1, 10, num=10),
        'R_mean': np.random.rand(5, 10)  # Dummy R_mean values
    }
    
    mu = np.array([0, np.pi/2, -np.pi-0.1, 2*np.pi])
    kappa = np.array([3, 11, 0, 5])
    expected_mu_idx = np.array([2, 3, 4, 2])
    expected_kappa_idx = np.array([3, 9, 0, 4])
    grid_vals, mu_idx, kappa_idx = interpolation.f_mu_kappa_grid_interp(mu=mu, kappa=kappa, 
                                                                        interp=interp, f='R_mean',
                                                                        return_idx=True,
                                                                        use_binary_search=use_binary_search)
    assert np.array_equal(mu_idx, expected_mu_idx)
    assert np.array_equal(kappa_idx, expected_kappa_idx)
    assert np.array_equal(interp['R_mean'][mu_idx, kappa_idx], grid_vals)
    assert len(grid_vals) == len(mu_idx)
    print("test_f_mu_kappa_grid_interp passed!")


# Test binary search methods
test_bs_vectorized()
test_find_closest_mu()
larger_test_find_closest_mu()
larger_test_find_closest_mu(num_mus=11, num_mu=250)
larger_test_find_closest_mu(num_mus=11, num_mu=1)

test_find_closest_kappa()
test_find_closest_mu()

R_mean_small = (np.arange(70)**2).reshape(7, 10)
grid_vals_small = np.array([R_mean_small[0, 3], R_mean_small[1, 3], R_mean_small[4, 3]])
interp_small = {
    'mus': np.array([-np.pi, -np.pi/2, -np.pi/3, 0, np.pi/3, np.pi/2, np.pi]),
    'kappas': np.linspace(0.1, 10, num=10), #[0.1,  1.2,  2.3,  3.4,  4.5,  5.6,  6.7,  7.8,  8.9, 10.]
    'R_mean': R_mean_small
}

for use_binary_search in [True, False]:
    fixed_kappa_test(mu=np.array([-np.pi, -np.pi/2+0.01, np.pi/3-0.02]), 
                    kappa=3,
                    interp=interp_small,
                    expected_mu_idx=[0, 1, 4],
                    expected_kappa_idx=3,
                    excepted_grid_vals=grid_vals_small,
                    use_binary_search=use_binary_search)
    test_f_mu_kappa_grid_interp(use_binary_search)

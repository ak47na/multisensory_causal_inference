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

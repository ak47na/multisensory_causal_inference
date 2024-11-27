import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import jax.numpy as jnp
import jax as jax

def initialize_data(data, EPS=1e-7):
    """
    Initialize and conditionally modify data based on grid boundaries.

    Parameters:
    - data: Dictionary containing 'grid' and 'pdf' as JAX arrays.
    - EPS: Float, epsilon value for boundary conditions.

    Returns:
    - Modified data dictionary with conditionally updated 'grid' and 'pdf'.
    """
    left_pdf_bound, right_pdf_bound = data['pdf'][0], data['pdf'][-1]

    # Condition for the left boundary
    condition_left = data['grid'][0] + jnp.pi > EPS

    # Conditionally set the first element of 'grid'
    new_grid_first = jax.lax.cond(
        condition_left,
        lambda _: -jnp.pi,
        lambda _: data['grid'][0],
        operand=None
    )

    # Conditionally set the first element of 'pdf'
    new_pdf_first = jax.lax.cond(
        condition_left,
        lambda _: left_pdf_bound,
        lambda _: data['pdf'][0],
        operand=None
    )

    # Update the first elements without changing the array shape
    data['grid'] = data['grid'].at[0].set(new_grid_first)
    data['pdf'] = data['pdf'].at[0].set(new_pdf_first)

    # Condition for the right boundary
    condition_right = data['grid'][-1] - jnp.pi < EPS

    # Conditionally set the last element of 'grid'
    new_grid_last = jax.lax.cond(
        condition_right,
        lambda _: jnp.pi,
        lambda _: data['grid'][-1],
        operand=None
    )

    # Conditionally set the last element of 'pdf'
    new_pdf_last = jax.lax.cond(
        condition_right,
        lambda _: right_pdf_bound,
        lambda _: data['pdf'][-1],
        operand=None
    )

    # Update the last elements without changing the array shape
    data['grid'] = data['grid'].at[-1].set(new_grid_last)
    data['pdf'] = data['pdf'].at[-1].set(new_pdf_last)

    return data

class UnifMap:    
    def __init__(self, data):
        self.data = data
        # left_pdf_bound, right_pdf_bound = self.data['pdf'][0], self.data['pdf'][-1]
        # if self.data['grid'][0] + jnp.pi > EPS:
        #     self.data['grid'] = jnp.insert(self.data['grid'], 0, -jnp.pi)
        #     self.data['pdf'] = jnp.insert(self.data['pdf'], 0, left_pdf_bound)
        # if self.data['grid'][-1] - np.pi < EPS:
        #     self.data['grid'] = jnp.append(self.data['grid'], jnp.pi)
        #     self.data['pdf'] = jnp.append(self.data['pdf'], right_pdf_bound)

    def interpolate_pdf(self):
        def interpolate_pdf_jax(self, x):
            return jnp.interp(x, self.data['grid'], self.data['pdf'], left=0.0, right=0.0)
        self.pdf_interp = interpolate_pdf_jax
        #self.pdf_interp = interp1d(self.data['grid'], self.data['pdf'], kind='linear', fill_value=0)

    def evaluate_pdf_at_samples(self, values):
        return self.pdf_interp(values)

    def evaluate_cdf_at_samples(self, points, linspace):
        """
        Interpolates the CDF values for given points in the linspace.

        Parameters:
        - cdf: An array of CDF values.
        - linspace: An array of points over which the CDF is defined.
        - points: An array of points at which to evaluate the CDF.

        Returns:
        - An array of interpolated CDF values at the specified points.
        """
        # Ensure the CDF array starts at 0 for proper interpolation
        corrected_cdf = self.cdf
        if self.cdf[0] != 0:
            corrected_cdf = jnp.insert(corrected_cdf, 0, 0)
            linspace = jnp.insert(linspace, 0, linspace[0] - (linspace[1] - linspace[0]))
        
        # Create an interpolation function based on the CDF and linspace
        # fill_value = (a, b) ensures CDF(x| x < linspace[0]) = a, CDF(x| x > linspace[-1]) = b
        cdf_func = interp1d(linspace, corrected_cdf, kind='linear', bounds_error=False, fill_value=(0, 1))
        
        # Evaluate the CDF at the given points
        cdf_at_points = cdf_func(points)
        return cdf_at_points

    def angle_space_to_unif_space(self, th):
        u_th = self.evaluate_cdf_at_samples(points=th, linspace=self.data['grid']) * jnp.pi * 2 - jnp.pi
        return u_th

    def unif_space_to_angle_space(self, u_th):
        th = self.inverse_cdf((u_th+jnp.pi)/(2*jnp.pi))
        return th

    def get_cdf_and_inverse_cdf(self, close_fig=True):
        # Compute the CDF
        cdf = jnp.cumsum(self.data['pdf']) * (self.data['grid'][1] - self.data['grid'][0])

        # Normalize the CDF
        cdf /= cdf[-1]
        #pdf_computed = jnp.diff(cdf, prepend=0) / (self.data['grid'][1] - self.data['grid'][0])

        # plt.plot(self.data['grid'], cdf)
        # plt.title('CDF for the shaping function')
        # plt.show()

        # Checking the CDF is correctly computed by determining the pdf from the cdf, then comparing with the shaping fn
        # plt.plot(self.data['grid'], pdf_computed)
        # plt.title('The PDF/the shaping')
        # plt.show()

        # Step 3: Interpolate the inverse CDF
        grid = self.data['grid']
        def inverse_cdf(q):
            """
            Compute the inverse CDF for given quantiles q.
            
            Parameters:
            - q: JAX array of quantiles in [0, 1].
            
            Returns:
            - Corresponding grid values for the quantiles.
            """
            return jnp.interp(q, cdf, grid, left=grid[0], right=grid[-1])
        #inverse_cdf = interp1d(cdf, self.data['grid'], kind='linear', bounds_error=False, fill_value=(self.data['grid'][0], self.data['grid'][-1]))

        # Example usage of the inverse CDF
        # probabilities = np.array([0.1, 0.5, 0.9])
        # x_values = inverse_cdf(probabilities)

        # print("X values for probabilities 0.1, 0.5, 0.9:", x_values)

        # Plot the inverse CDF
        #plt.figure(figsize=(8, 6))
        p = jnp.linspace(0, 1, 1000)
        # plt.plot(p, inverse_cdf(p))
        # plt.xlabel('Probability')
        # plt.ylabel('X value')
        # plt.title('Inverse CDF')
        # plt.grid(True)
        # plt.show()
        self.cdf = cdf
        self.inverse_cdf = inverse_cdf
        self.interpolate_pdf()
        #grid_mus = jnp.linspace(-jnp.pi, jnp.pi, num=250)
        # plt.plot(grid_mus, self.evaluate_pdf_at_samples(grid_mus))
        # plt.title('Interpolated pdf')
        # plt.show()

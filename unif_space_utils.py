import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class UnifMap:
    def __init__(self, data):
        self.data = data
        # note: grid adapted for (-pi/2, pi/2] range
        self.data['grid'] = self.data['grid']/2.0

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
            corrected_cdf = np.insert(corrected_cdf, 0, 0)
            linspace = np.insert(linspace, 0, linspace[0] - (linspace[1] - linspace[0]))
        
        # Create an interpolation function based on the CDF and linspace
        cdf_func = interp1d(linspace, corrected_cdf, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Evaluate the CDF at the given points
        cdf_at_points = cdf_func(points)
        return cdf_at_points

    def angle_space_to_unif_space(self, th):
        # note: grid adapted for (-pi/2, pi/2] range
        u_th = self.evaluate_cdf_at_samples(points=th, linspace=self.data['grid']) * np.pi - np.pi/2.0
        return u_th

    def unif_space_to_angle_space(self, u_th):
        # note: grid adapted for (-pi/2, pi/2] range
        th = self.inverse_cdf((u_th+np.pi/2.0)/(np.pi))
        return th

    def get_cdf_and_inverse_cdf(self, plotting=False):
        # Compute the CDF
        cdf = np.cumsum(self.data['pdf']) * (self.data['grid'][1] - self.data['grid'][0])

        # Normalize the CDF
        cdf /= cdf[-1]
        pdf_computed = np.diff(cdf, prepend=0) / (self.data['grid'][1] - self.data['grid'][0])

        if plotting == True:
            plt.plot(self.data['grid'], cdf)
            plt.title('CDF for the shaping function')
            plt.show()

            # Checking the CDF is correctly computed by determining the pdf from the cdf, then comparing with the shaping fn
            plt.plot(self.data['grid'], pdf_computed)
            plt.title('The PDF/the shaping')
            plt.show()

        # Step 3: Interpolate the inverse CDF
        inverse_cdf = interp1d(cdf, self.data['grid'], kind='linear', bounds_error=False, fill_value="extrapolate")

        # Example usage of the inverse CDF
        # probabilities = np.array([0.1, 0.5, 0.9])
        # x_values = inverse_cdf(probabilities)

        # print("X values for probabilities 0.1, 0.5, 0.9:", x_values)

        if plotting == True:
            # Plot the inverse CDF
            plt.figure(figsize=(8, 6))
            p = np.linspace(0, 1, 1000)
            plt.plot(p, inverse_cdf(p))
            plt.xlabel('Probability')
            plt.ylabel('X value')
            plt.title('Inverse CDF')
            plt.grid(True)
            plt.show()
        self.cdf = cdf
        self.inverse_cdf = inverse_cdf
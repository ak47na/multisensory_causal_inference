import abc
import matplotlib.pyplot as plt

class CausalInference(abc.ABC):
    def __init__(self, distribution='gaussian'):
        self.distribution = distribution

    @abc.abstractmethod
    def segregation_estimate(self, x, mu_p, sigma, sigma_p):
        pass

    @abc.abstractmethod
    def fusion_estimate(self, x_a, x_v, sigma_a, sigma_v, mu_p, sigma_p, return_sigma=False):
        pass

    @abc.abstractmethod
    def likelihood_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        pass

    @abc.abstractmethod
    def likelihood_separate_causes(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
        pass

    @abc.abstractmethod
    def posterior_prob_common_cause(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass

    @abc.abstractmethod
    def bayesian_causal_inference(self, x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
        pass


def plot_histograms(a, b, x, y, x_min, x_max, filepath, a_label='Vis', b_label='Aud'):
    _, x_size, y_size = a.shape
    
    # Create a grid of subplots
    fig, axs = plt.subplots(x_size, y_size, figsize=(12, 8))
    
    # Flatten the subplots array to iterate over each subplot
    axs_flat = axs.flatten()
    
    # Iterate over each pair (x, y) and plot the histogram of data points in arrays a and b
    for i in range(x_size):
        for j in range(y_size):
            # Get the data points for the current pair (x, y)
            data_a = a[:, i, j].flatten()
            data_b = b[:, i, j].flatten()
            
            # Plot the histograms
            axs_flat[i * y_size + j].hist(data_a, bins=20, label=a_label, alpha=0.5, edgecolor='b', histtype='step', density=True)
            axs_flat[i * y_size + j].hist(data_b, bins=20, label=b_label, alpha=0.5, edgecolor='r', histtype='step', density=True)
            
            # Add labels and legend
            axs_flat[i * y_size + j].set_xlim(x_min, x_max)
            axs_flat[i * y_size + j].set_title(f'{x}={i}, {y}={j}')
            axs_flat[i * y_size + j].legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(filepath)
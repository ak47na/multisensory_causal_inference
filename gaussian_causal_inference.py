import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def segregation_estimate(x, mu_p, sigma, sigma_p):
    """
    Compute the MAP estimate in the segregation case (independent sensory processing).
    The posterior p(s|x) \propto p(x|s)p(s), with p(s) as the prior and p(x|s) as the likelihood.

    Parameters:
    x (float or np.ndarray): Sensory input (e.g., observed rate).
    mu_p (float or np.ndarray): Prior mean of the stimulus rate.
    sigma (float): Sensory noise (standard deviation).
    sigma_p (float): Prior standard deviation of the stimulus rate.

    Returns:
    (float or np.ndarray): Segregation estimate matching the type of x.
    """
    return (x * sigma_p**2 + mu_p * sigma**2) / (sigma**2 + sigma_p**2)

def fusion_estimate(x_a, x_v, sigma_a, sigma_v, mu_p, sigma_p, return_sigma=False):
    """
    Compute the MAP estimate in the fusion case (combined sensory processing).
    The posterior p(s|x_a, x_v) \propto p(x_a, x_v|s)p(s) = p(x_a|s)p(x_v|s)p(s) with p(s) as the
    prior and p(x_a|s), p(x_v|s) as the likelihood.

    Parameters:
    x_a (float or np.ndarray): Auditory sensory input (observed rate).
    x_v (float or np.ndarray): Visual sensory input (observed rate).
    sigma_a (float): Auditory sensory noise (standard deviation).
    sigma_v (float): Visual sensory noise (standard deviation).
    mu_p (float or np.ndarray): Prior mean of the stimulus rate.
    sigma_p (float): Prior standard deviation of the stimulus rate.

    Returns:
    (float or np.ndarray): Fusion estimate.
    """
    num = x_a * ((sigma_v*sigma_p)**2) + x_v * ((sigma_a*sigma_p)**2) + mu_p * ((sigma_v*sigma_a)**2)
    denom = ((sigma_v*sigma_p)**2) + ((sigma_a*sigma_p)**2) + ((sigma_v*sigma_a)**2)
    if return_sigma:
        return num / denom, 1/np.sqrt((1/sigma_a**2)+(1/sigma_v**2)+(1/sigma_p**2))
    return num / denom

def likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
    """
    Compute the likelihood of the common cause hypothesis (signals from the same source).
    p(x_v, x_a|C=1) = \int p(x_v, x_a|s)p(s)ds = \int p(x_v|s)p(x_a|s)p(s)ds

    Parameters:
    x_v (float or np.ndarray): Visual sensory input (observed rate).
    x_a (float or np.ndarray): Auditory sensory input (observed rate).
    sigma_v (float): Visual sensory noise (standard deviation).
    sigma_a (float): Auditory sensory noise (standard deviation).
    mu_p (float or np.ndarray): Prior mean of the stimulus rate.
    sigma_p (float): Prior standard deviation of the stimulus rate.

    Returns:
    (float or np.ndarray): Likelihood of the common cause hypothesis.
    """
    var_common = sigma_v**2 * sigma_a**2 + sigma_v**2 * sigma_p**2 + sigma_a**2 * sigma_p**2
    exp_common = ((x_v - x_a)**2 * sigma_p**2 + (x_v - mu_p)**2 * sigma_a**2 + (x_a - mu_p)**2 * sigma_v**2) / (2 * var_common)
    return np.exp(-exp_common) / (2 * np.pi * np.sqrt(var_common))

def likelihood_separate_causes(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p):
    """
    Compute the likelihood of the separate causes hypothesis (signals from different sources).
    p(x_v, x_a|C=2) = \int \int p(x_v, x_a|s_v, s_a)p(s_v)p(s_a)ds_v ds_a and due to independence
                    = (\int p(x_v|s_v)p(s_v) ds_v)(\int p(x_v|s_v)p(s_v) ds_a)

    Parameters:
    x_v (float or np.ndarray): Visual sensory input (observed rate).
    x_a (float or np.ndarray): Auditory sensory input (observed rate).
    sigma_v (float): Visual sensory noise (standard deviation).
    sigma_a (float): Auditory sensory noise (standard deviation).
    mu_p (float or np.ndarray): Prior mean of the stimulus rate.
    sigma_p (float): Prior standard deviation of the stimulus rate.

    Returns:
    (float or np.ndarray): Likelihood of the separate causes hypothesis.
    """
    var_sep_v = sigma_v**2 + sigma_p**2
    var_sep_a = sigma_a**2 + sigma_p**2
    exp_sep = ((x_v - mu_p)**2 / (2 * var_sep_v)) + ((x_a - mu_p)**2 / (2 * var_sep_a))
    return np.exp(-exp_sep) / (2 * np.pi * np.sqrt(var_sep_v * var_sep_a))

def posterior_prob_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
    """
    Compute the posterior probability of the common cause hypothesis.
    By Bayes rule P(C=1|x_v, x_a) = \frac{P(x_v, x_a| C=1) p(C=1)}{P(x_v, x_a)}
    P(C=1|x_v, x_a) = \frac{P(x_v, x_a| C=1) p(C=1)}{P(x_v, x_a|C=1)P(C=1) + P(x_v, x_a|C=2)P(C=2)}
    Here P(C=1) = p_common = pi_c.

    Parameters:
    x_v (float or np.ndarray): Visual sensory input (observed rate).
    x_a (float or np.ndarray): Auditory sensory input (observed rate).
    sigma_v (float): Visual sensory noise (standard deviation).
    sigma_a (float): Auditory sensory noise (standard deviation).
    mu_p (float or np.ndarray): Prior mean of the stimulus rate.
    sigma_p (float): Prior standard deviation of the stimulus rate.
    pi_c (float or np.ndarray): Prior probability of the common cause hypothesis.

    Returns:
    float or np.ndarray: Posterior probability of the common cause hypothesis.
    """
    posterior_p_common = likelihood_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * pi_c
    posterior_p_separate = likelihood_separate_causes(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p) * (1 - pi_c)
    return posterior_p_common / (posterior_p_common + posterior_p_separate)

def bayesian_causal_inference(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c):
    """
    Compute the Bayesian causal inference estimate.

    Parameters:
    x_v (float): Visual sensory input (observed rate).
    x_a (float): Auditory sensory input (observed rate).
    sigma_v (float): Visual sensory noise (standard deviation).
    sigma_a (float): Auditory sensory noise (standard deviation).
    mu_p (float): Prior mean of the stimulus rate.
    sigma_p (float): Prior standard deviation of the stimulus rate.
    pi_c (float): Prior probability of the common cause hypothesis.

    Returns:
    float: Bayesian causal inference estimate.
    """
    # Compute the posterior probability of the common cause hypothesis: P(C=1|x_v, x_a)
    posterior_p_common = posterior_prob_common_cause(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p, pi_c)
    # Compute the segregated estimate (use the more reliable modality)
    segregated_estimate_v = segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
    segregated_estimate_a = segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
    # Compute the fused estimate (combined sensory information)
    fused_estimate = fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
    # Return the weighted average of the segregated and fused estimates based on the posterior probability
    s_v_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_v
    s_a_hat = posterior_p_common * fused_estimate + (1 - posterior_p_common) * segregated_estimate_a
    return s_v_hat, s_a_hat

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

if __name__ == "__main__()":
    # Data analysis figure2: "we simulate the system for each combination of cues for 10,000 times"
    num_sim = 10000
    stimuli_values = np.linspace(-10, 10, 5)
    s_vs, s_as = np.meshgrid(stimuli_values, stimuli_values, indexing='ij')

    sigma_v, sigma_a = 2.14, 9.2  # Sensory noise for visual and auditory inputs (9.2+-1.1)
    mu_p, sigma_p = 0, 12.3  # Prior mean and standard deviation for the stimulus rate
    pi_c = 0.23  # Prior probability of the common cause hypothesis
    print(f'Svs = {s_vs.reshape(-1)}\nSas = {s_as.reshape(-1)}\n')

    # Generate random samples for each combination of cues
    x_v = norm.rvs(loc=s_vs, scale=sigma_v, size=(num_sim, stimuli_values.size, stimuli_values.size))
    x_a = norm.rvs(loc=s_as, scale=sigma_a, size=(num_sim, stimuli_values.size, stimuli_values.size))

    # Compute the estimates using the defined functions
    segregated_est_v = segregation_estimate(x_v, mu_p, sigma_v, sigma_p)
    segregated_est_a = segregation_estimate(x_a, mu_p, sigma_a, sigma_p)
    fused_est = fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
    causal_inference_est_v, causal_inference_est_a = bayesian_causal_inference(x_v, x_a, sigma_v,
                                                                    sigma_a, mu_p, sigma_p, pi_c)

    print(f'segregated_est_v={segregated_est_v.mean(axis=0)}\n')
    print(f'segregated_est_a={segregated_est_a.mean(axis=0)}\n')
    print(f'fused_est={fused_est.mean(axis=0)}\n')
    print(f'causal_inference_est_v={causal_inference_est_v.mean(axis=0)}\n')
    print(f'causal_inference_est_a={causal_inference_est_a.mean(axis=0)}\n')

    plot_histograms(causal_inference_est_v, causal_inference_est_a, x='Aud', y='Vis', 
                    x_min=stimuli_values[0], x_max=stimuli_values[-1], filepath='./fig2c.png')

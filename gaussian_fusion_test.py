import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import causal_inference
import gaussian_causal_inference

num_sim = 1000
stimuli_values = np.linspace(-10, 10, 5)
s_vs, s_as = np.meshgrid(stimuli_values, stimuli_values, indexing='ij')

sigma_v, sigma_a = 2.14, 9.2  # Sensory noise for visual and auditory inputs (9.2+-1.1)
mu_p, sigma_p = 0, 12.3  # Prior mean and standard deviation for the stimulus rate
pi_c = 0.23  # Prior probability of the common cause hypothesis
print(f'Svs = {s_vs.reshape(-1)}\nSas = {s_as.reshape(-1)}\n')

# Generate random samples for each combination of cues
x_v = norm.rvs(loc=s_vs, scale=sigma_v, size=(num_sim, stimuli_values.size, stimuli_values.size))
x_a = norm.rvs(loc=s_as, scale=sigma_a, size=(num_sim, stimuli_values.size, stimuli_values.size))

model = gaussian_causal_inference.GaussianCausalInference()
sim_model = gaussian_causal_inference.GaussianCausalInference(simulate=False)
fused_est = sim_model.fusion_estimate(x_v, x_a, sigma_v, sigma_a, mu_p, sigma_p)
fused_est_mu, fused_est_sigma  = model.fusion_posterior_params(s_a=s_as, s_v=s_vs, 
                                                                           sigma_a=sigma_a, 
                                                                           sigma_v=sigma_v, 
                                                                           mu_p=mu_p, 
                                                                           sigma_p=sigma_p)
fused_est_analytic = norm.rvs(loc=fused_est_mu, scale=fused_est_sigma,
                            size=(num_sim, stimuli_values.size, stimuli_values.size))
data_a = fused_est_analytic[:, 1,1]
data_b = fused_est[:, 1,1]
plt.hist(data_a, bins=20, label='analytic', alpha=0.5, edgecolor='b', histtype='step', density=True)
plt.hist(data_b, bins=20, label='sim', alpha=0.5, edgecolor='r', histtype='step', density=True)
plt.legend()
plt.show()

causal_inference.plot_histograms(a=fused_est_analytic, b=fused_est,
                                          x='Aud', y='Vis', 
                                          x_min=stimuli_values[0], 
                                          x_max=stimuli_values[-1], 
                                          a_label='analytic',
                                          b_label='sim',
                                          filepath='./test_fuesd_est.png')
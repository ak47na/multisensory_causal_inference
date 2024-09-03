import numpy as np
import pickle
import uniformised_space_utils as usu
from scipy.stats import vonmises, circmean
import matplotlib.pyplot as plt
from custom_causal_inference import CustomCausalInference
from repulsion_hypothesis import repulsion_value
import utils


def heatmap_f_s_n_t(f_s_n_t, s_n, t, f_name):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(s_n[0], t[:,0], f_s_n_t, shading='auto', cmap='RdBu_r')
    plt.colorbar(label=f_name)
    plt.xlabel('s_n')
    plt.ylabel('t')
    plt.title(f'Heatmap of {f_name} with s_n and t')
    plt.show()
    plt.clf()

def plot_estimate(us_n, ut, r_n, mu_x_dim, estimate, est_label, plot_heatmap):
    plt.scatter(us_n, ut, label='u_t', alpha=.5)
    plt.scatter(us_n, r_n, label='r_n', alpha=.5)
    if estimate.ndim > 1:
        for i in range(estimate.shape[1]):
            plt.scatter(us_n, estimate[:,i], label=f'{est_label}_{i}', alpha=.5)
    else:
        plt.scatter(us_n, estimate, label=est_label, alpha=.5)
    plt.title(f'Values for {est_label}')
    plt.xlabel('us_n')
    plt.legend()
    plt.show()
    plt.clf()
    if plot_heatmap:
        uts_unq = ut.reshape(mu_x_dim,mu_x_dim)[:,0]
        if estimate.ndim > 1:
            for i in range(estimate.shape[1]):
                plt.pcolormesh(us_n[:mu_x_dim], uts_unq, 
                                estimate[:,i].reshape(mu_x_dim,mu_x_dim), 
                                shading='auto', cmap='RdBu_r')
                plt.colorbar(label=est_label)
                plt.title(f'Heatmap for {est_label}_{i}')
                plt.xlabel('us_n')
                plt.ylabel('ut')
                plt.show()
                plt.clf()

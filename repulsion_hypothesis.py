import numpy as np
import causal_inference
import uniformised_space_utils as usu
import pickle
import utils
from scipy.stats import vonmises, circmean, mode
from scipy.special import i0
import matplotlib.pyplot as plt
from custom_causal_inference import CustomCausalInference

def repulsion_value(t, s_n, r):
    d1 = utils.circular_dist(s_n, t)
    d2 = utils.circular_dist(r, t)
    d3 = utils.circular_dist(s_n, r)
    return d2+d3-d1


def test_repulsion_fixed_kappas(ut, us_n, r_n, kappa1, kappa2, decision_rule, p_commons, num_sim=1000):
    model = CustomCausalInference(decision_rule=decision_rule)
    t_samples = vonmises(loc=ut, kappa=kappa1).rvs(size=(num_sim, us_n.shape[0]))
    s_n_samples = vonmises(loc=us_n, kappa=kappa2).rvs(size=(num_sim, us_n.shape[0]))
    responses = []
    indices_to_plot = np.round(np.linspace(0, len(ut)-1, 10)).astype(int)
    for p_common in p_commons:
        responses.append(model.bayesian_causal_inference(x_v=t_samples, x_a=s_n_samples, 
                                                    sigma_v=kappa1, sigma_a=kappa2,
                                                    mu_p=None, sigma_p=None,
                                                    pi_c=p_common))
        for idx in indices_to_plot:
            plt.hist(responses[-1][0][:, idx], bins=65, label='t est', alpha=0.5, edgecolor='r', density=True, histtype='step')
            plt.hist(responses[-1][1][:, idx], bins=65, label='s_n est', alpha=0.5, edgecolor='r', density=True, histtype='step')
            plt.axvline(x=r_n[idx], color='g', linestyle='--', linewidth=2, label='r_n')
            plt.axvline(x=us_n[idx], color='b', linestyle='--', linewidth=2, label=f'us_n')
            plt.axvline(x=ut[idx], color='y', linestyle='--', linewidth=2, label=f'ut')
            plt.legend()
            plt.title(f'Von Mises approximation and simulated distribution of {decision_rule} responses t, s_n={ut[idx], us_n[idx]}')
            plt.show()


    # plt.hist(fused_est[:, idx1, idx2], bins=65, label='fused est', alpha=0.5, edgecolor='r', density=True, histtype='step')
    # plt.hist(segregated_est_s_n[:, idx1, idx2], bins=65, label='seggregated est', alpha=0.5, edgecolor='b', density=True, histtype='step')
    # plt.axvline(x=mean_fused_est[idx1, idx2], color='k', linestyle='--', linewidth=2, label='mean fusion est')
    # plt.axvline(x=r_n[idx1, idx2], color='g', linestyle='--', linewidth=2, label='r_n')
    # plt.axvline(x=s_n[idx1, idx2], color='b', linestyle='--', linewidth=2, label=f's_n')
    # plt.axvline(x=t[idx1, idx2], color='y', linestyle='--', linewidth=2, label=f't')
    # plt.axvline(x=mode_fused_est[idx1, idx2], color='m', linestyle='--', linewidth=2, label=f'mode fusion est, c={count_mode[idx1, idx2]}')
    # plt.legend()
    # plt.title(f'Von Mises approximation and simulated distribution of mean responses t, s_n={t[idx1, idx2], s_n[idx1, idx2]}')
    # plt.show()
    # # import pdb; pdb.set_trace()
    # # mode_fused_est = fused_est.mode(axis=0)
    # rep_mode = repulsion_value(t=t, s_n=s_n, r=mode_fused_est)
    # rep_mean = repulsion_value(t=t, s_n=s_n, r=mean_fused_est)
    # rep_rn = repulsion_value(t=t, s_n=s_n, r=r_n)
    # d = utils.circular_dist(mode_fused_est[idx1, idx2], r_n[idx1, idx2])
    # print(f'max rep_mode={rep_mode.max()}, rep_mean={rep_mean.max()}, rep_rn={rep_rn.max()}')
    # import pdb; pdb.set_trace()


def plot_task_var(task_var, task_var_names):
    _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_var)))

    # Plot angles and their reflections
    for i, (angle_i, label_i) in enumerate(zip(task_var, task_var_names)):
        ax.plot([angle_i], [1], 'o', color=colors[i], label=label_i)


    # Draw quadrant lines
    ax.plot([0, 0], [0, 1], 'k--')
    ax.plot([np.pi/2, np.pi/2], [0, 1], 'k--')
    ax.plot([np.pi, np.pi], [0, 1], 'k--')
    ax.plot([-np.pi/2, -np.pi/2], [0, 1], 'k--')

    # Set theta direction and zero location
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_thetamin(-180)
    ax.set_thetamax(180)

    # Set labels and title
    ax.set_xticks([0, np.pi/2, np.pi, -np.pi/2])
    ax.set_xticklabels(['0', 'π/2', '-π', '-π/2'])
    ax.set_yticklabels([])
    ax.set_title('Task variables')
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
    #plt.legend()
    plt.show()


def get_s_n_and_t(grid, gam_data):
    all_sns, all_ts = [], []
    r_ns = []
    for i, t in enumerate(grid[10:-10]):
        for j, s_n in enumerate(grid[100:-10]):
            r_n = gam_data['full_pdf_mat'][i, j, 2]
            d1 = utils.circular_dist(s_n, t)
            d2 = utils.circular_dist(r_n, t)
            d3 = utils.circular_dist(r_n, s_n)
            if (d2+d3-d1 > 0.1) and (utils.circular_dist(s_n, t) > np.pi-0.1):
                # print(f'd={d2+d3-d1}, d1,d2,d2={d1, d2, d3}, r_n={r_n}, s_n={s_n}, t={t}')
                all_sns.append(s_n)
                all_ts.append(t)
                r_ns.append(r_n)
    return np.array(all_sns), np.array(all_ts), np.array(r_ns)

if __name__ == "__main__":
    num_sim = 10000
    D = 250  # grid dimension 
    angle_gam_data_path = 'D:/AK_Q1_2024/Gatsby/data/base_bayesian_contour_1_circular_gam/base_bayesian_contour_1_circular_gam.pkl'
    unif_fn_data_path='D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl'
    # Load GAM
    with open(angle_gam_data_path, 'rb') as file:
        gam_data = pickle.load(file)
    with open(unif_fn_data_path, 'rb') as file:
            unif_fn_data = pickle.load(file)
    unif_map = usu.UnifMap(data=unif_fn_data)
    unif_map.get_cdf_and_inverse_cdf()
    map_to_unif_space = True
    grid = np.linspace(-np.pi, np.pi, num=D)
    s_n, t, r_n = get_s_n_and_t(grid, gam_data)
    if map_to_unif_space:
        s_n = unif_map.angle_space_to_unif_space(s_n)
        t = unif_map.angle_space_to_unif_space(t)
    print(f'Running cue combination for t={t}, s_n={s_n}, r_n={r_n}')
    kappa1, kappa2 = 250, 250
    for decision_rule in ['mean', 'mode']:
        test_repulsion_fixed_kappas(ut=t, us_n=s_n, r_n=r_n, kappa1=kappa1, kappa2=kappa2, 
                                    num_sim=num_sim, decision_rule=decision_rule, p_commons=[0, .2, .5])

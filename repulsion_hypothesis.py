import numpy as np
import uniformised_space_utils as usu
import pickle
import utils
from scipy.stats import vonmises, circmean, mode
import matplotlib.pyplot as plt
from custom_causal_inference import CustomCausalInference

def repulsion_value(t, s_n, r):
    d1 = utils.circular_dist(s_n, t)
    d2 = utils.circular_dist(r, t)
    d3 = utils.circular_dist(s_n, r)
    return d2+d3-d1


def test_repulsion_fixed_kappas(t, s_n, ut, us_n, r_n, kappa1, kappa2, decision_rule, p_commons, 
                                num_sim=1000):
    model = CustomCausalInference(decision_rule=decision_rule)
    # Samples in internal spaceare assumed to beVon Mises.
    t_samples = vonmises(loc=ut, kappa=kappa1).rvs(size=(num_sim, us_n.shape[0]))
    s_n_samples = vonmises(loc=us_n, kappa=kappa2).rvs(size=(num_sim, us_n.shape[0]))
    responses = []
    for pc_idx, p_common in enumerate(p_commons):
        # Add "optimal" estimates for s_n and t for every pair of samples assuming P(C=1)=p_common.
        responses.append(model.bayesian_causal_inference(x_v=t_samples, x_a=s_n_samples, 
                                                    sigma_v=kappa1, sigma_a=kappa2,
                                                    mu_p=None, sigma_p=None,
                                                    pi_c=p_common))
        # Find circular mean across "optimal" estimates for samples.
        mean_t_est = circmean(unif_map.unif_space_to_angle_space(responses[-1][0]), low=-np.pi, high=np.pi, axis=0)
        mean_sn_est = circmean(unif_map.unif_space_to_angle_space(responses[-1][1]), low=-np.pi, high=np.pi, axis=0)
        plt.plot(mean_t_est, 'r', linestyle='--', label='t_est')
        plt.plot(mean_sn_est, 'b', linestyle='--', label='sn_est')
        plt.plot(r_n, color='g', label='r_n')
        plt.plot(us_n, color='b', label=f'us_n')
        plt.plot(ut, color='r', label=f'ut')
        plt.legend()
        plt.title(f'Mean estimates for {decision_rule} responses, p_c={p_common}')
        plt.savefig(f'./plots/causal_estim_{decision_rule}_{pc_idx}.png')
        plt.clf()
        # Plot the repulsion between circmean of "optimal" estimates and cue means.
        repuslion_t = repulsion_value(t=t,s_n=s_n, r=mean_t_est)
        repuslion_sn = repulsion_value(t=t,s_n=s_n, r=mean_sn_est)
        plt.scatter(repuslion_t, 'r', label='repulsion t est', alpha=.4)
        plt.scatter(repuslion_sn, 'b', label='repulsion sn est', alpha=.4)
        plt.scatter(repulsion_value(t=t, s_n=s_n, r=r_n), color='g', label='repulsion r_n', alpha=.4)
        #plt.plot(s_n, color='b', label=f's_n')
        #plt.plot(t, color='r', label=f't')
        plt.legend()
        plt.title(f'Repulsion for estimates using {decision_rule} responses')
        plt.savefig(f'./plots/repulsion_{decision_rule}_{pc_idx}.png')
        plt.clf()
        # Plot the distribution of "optimal" estimates for 10 uniform (u)s_n, (u)t pairs.
        indices_to_plot = np.arange(0, len(s_n), step=len(s_n)//10, dtype=int)
        for idx in indices_to_plot:
            plt.hist(responses[-1][0][:, idx], bins=65, label='t est', alpha=0.5, edgecolor='r', density=True, histtype='step')
            plt.hist(responses[-1][1][:, idx], bins=65, label='s_n est', alpha=0.5, edgecolor='m', density=True, histtype='step')
            plt.axvline(x=r_n[idx], color='g', linestyle='--', linewidth=2, label='r_n')
            plt.axvline(x=us_n[idx], color='b', linestyle='--', linewidth=2, label=f'us_n')
            plt.axvline(x=ut[idx], color='y', linestyle='--', linewidth=2, label=f'ut')
            plt.legend()
            plt.title(f'Von Mises approximation and simulated distribution of {decision_rule} responses ut, us_n={ut[idx], us_n[idx]}')
            plt.savefig(f'./plots/hist_responses_{decision_rule}_{idx}.png')
            plt.clf()


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
    for i, t in enumerate(grid):
        for j, s_n in enumerate(grid):
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
    # Load the GAM.
    with open(angle_gam_data_path, 'rb') as file:
        gam_data = pickle.load(file)
    # Load the uniformising function data.
    with open(unif_fn_data_path, 'rb') as file:
            unif_fn_data = pickle.load(file)
    # Initialise uniformising function map.
    unif_map = usu.UnifMap(data=unif_fn_data)
    unif_map.get_cdf_and_inverse_cdf()

    grid = np.linspace(-np.pi, np.pi, num=D)
    s_n, t, r_n = get_s_n_and_t(grid, gam_data)
    plt.scatter(s_n, t, label='t')
    plt.scatter(s_n, r_n, label='r_n')
    plt.legend()
    plt.savefig('./plots/selected_means.png')
    plt.clf()
    # Map means to uniformised space.
    us_n = unif_map.angle_space_to_unif_space(s_n)
    ut = unif_map.angle_space_to_unif_space(t)
    print(f'Running causal cue combination for t={t.shape}, s_n={s_n.shape}, r_n={r_n.shape}')
    kappa1, kappa2 = 250, 250
    for decision_rule in ['mode', 'mean']:
        test_repulsion_fixed_kappas(t, s_n, ut=ut, us_n=us_n, r_n=r_n, kappa1=kappa1, kappa2=kappa2, 
                                    num_sim=num_sim, decision_rule=decision_rule, p_commons=[0, .2, .5])

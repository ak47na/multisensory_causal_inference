from abc import ABC, abstractmethod
import utils
import numpy as np
import uniformised_space_utils as usu
from scipy.stats import vonmises, circmean 
import matplotlib.pyplot as plt
import pickle
import interpolation


class Distribution(ABC):
    def __init__(self, loc, scale, kappa=None, interp=None) -> None:
        self.loc = loc
        self.scale = scale
        self.kappa = kappa
        self.interp = interp
        
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def mode(self):
        pass

class UVM(Distribution):
    def __init__(self, loc, scale, kappa, interp=None, num_sim=1000, 
                 unif_fn_data_path='D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl') -> None:
        super().__init__(loc, scale, kappa, interp)
        #assert (np.asarray(loc).shape == np.asarray(kappa).shape)
        self.num_sim = num_sim
        self.interp = interp
        if interp is None:
            self.load_interp()
        with open(unif_fn_data_path, 'rb') as file:
            unif_fn_data = pickle.load(file)
        self.unif_map = usu.UnifMap(data=unif_fn_data)
        self.unif_map.get_cdf_and_inverse_cdf()

    def load_interp(self, mean_file_path='./learned_data/R_mean_250_250.npy',
                    mode_file_path='./learned_data/R_mode_250_250.npy'):
        mus = np.linspace(-np.pi, np.pi, 250)
        kappas = np.linspace(1, 500, 250)
        self.interp = {'mus': mus, 'kappas': kappas}
        self.interp['R_mean'] = np.load(mean_file_path)
        self.interp['R_mode'] = np.load(mode_file_path)

    def mean(self):
        return interpolation.f_mu_kappa_grid_interp(mu=self.loc,
                                             kappa=self.kappa, 
                                             interp=self.interp, 
                                             f='R_mean',
                                             use_binary_search=False)
    
    def learn_mean_and_mode(self, mean_file_path=None, mode_file_path=None):
        if isinstance(self.loc, int):
            sample_size = self.num_sim
        else:
            sample_size = (self.num_sim, *self.loc.shape)
        samples = self.rvs(size=sample_size)
        
        self.interp['R_mean'] = circmean(samples, low=-np.pi, high=np.pi, axis=0)
        self.interp['R_mode'] = utils.modes(samples, num_bins=250)
        if mean_file_path is not None:
            np.save(mean_file_path, self.interp['R_mean'])
        if mode_file_path is not None:
            np.save(mode_file_path, self.interp['R_mode'])
    
    def mode(self):
        return interpolation.f_mu_kappa_grid_interp(mu=self.loc,
                                             kappa=self.kappa, 
                                             interp=self.interp, 
                                             f='R_mode',
                                             use_binary_search=False)
        
    
    def decision_rule(self, decision_rule):
        if decision_rule == 'mean':
            return self.mean()
        return self.mode()
    
    def rvs(self, size):
        if isinstance(size, int):
            assert (isinstance(self.loc, float) and isinstance(self.kappa, float))
        print(f'Generating samples {size} for UVM(loc={self.loc.shape}, kappa={self.kappa.shape})')
        samples = utils.wrap(vonmises.rvs(kappa=self.kappa, loc=self.loc, size=size))
        print(f'Converting samples {size} for UVM(loc={self.loc.shape}, kappa={self.kappa.shape})')
        samples = self.unif_map.unif_space_to_angle_space(samples)
        return samples

    def plot_decision_rules(self, decision_rules, filepath='./plots/decision_rules'):
        col = ['r', 'b']
        if len(decision_rules)==1:
            col = ['b']
        for decision_rule, color in zip(decision_rules, col):
            for i, _ in enumerate(self.kappa):
                label = None
                if i == 0:
                    label = decision_rule
                plt.scatter(self.loc[:, i], self.interp[f'R_{decision_rule}'][:, i], c=color, label=label, alpha=.2)
        plt.legend()
        plt.savefig(filepath)
        plt.clf()


if __name__ == "__main__":
    mus = np.linspace(-np.pi, np.pi, 250)
    kappas = np.linspace(1, 500, 250)
    mus_matrix = np.tile(mus[:, np.newaxis], (1, len(kappas)))
    kappas_matrix = np.tile(kappas, (len(mus), 1))
    interp = {'mus': mus, 'kappas': kappas}
    num_sim = 1000
    uvm = UVM(loc=mus_matrix, scale=None, kappa=kappas_matrix, interp=interp, num_sim=num_sim, 
              unif_fn_data_path='D:/AK_Q1_2024/Gatsby/uniform_model_base_inv_kappa_free.pkl')
    uvm.learn_mean_and_mode(mean_file_path=f'./learned_data/R_mean_{len(mus)}_{len(kappas)}.npy',
                            mode_file_path=f'./learned_data/R_mode_{len(mus)}_{len(kappas)}.npy')
    uvm.plot_decision_rules(decision_rules=['mean', 'mode'])

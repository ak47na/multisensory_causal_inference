from abc import ABC, abstractmethod
import utils


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
    def __init__(self, loc, scale, kappa, interp, num_sim) -> None:
        super().__init__(loc, scale, kappa, interp)
        self.num_sim = num_sim
        self.unif_map = usu.UnifMap(data=unif_fn_data)
        unif_map.get_cdf_and_inverse_cdf()

    def mean(self):
        return self.interp.get('R_mean')
    
    def learn_mean_and_mode(self, mean_file_path, mode_file_path):
        if isinstance(self.loc, int):
            sample_size = self.num_sim
        else:
            sample_size = (self.num_sim, *loc.shape)
        samples = utils.wrap(vonmises.rvs(kappa=self.kappa, loc=self.loc, size=sample_size))
        samples = self.unif_map.unif_space_to_angle_space(samples)
        np.save(circmean(samples, low=-np.pi, high=np.pi, axis=0), mean_file_path)
        np.save(utils.mode(samples), mode_file_path)
    
    def mode(self):
        return self.interp.get('R_mode')
    
    # def rvs(self, x, size):
    #     if isinstance(size, int):
    #         assert (isinstance(self.loc, float) and isinstance(self.kappa, float))
    #     else:
    #         # use chatGPT to see existing solution to sample from U^{-1}[VM], i.e. custom dist
    
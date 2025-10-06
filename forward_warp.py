import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d, circmean
# from load import load_online
import uniformised_space_utils as usu
from tqdm import tqdm
from prev_resp import model_plot
from von_mises_causal_inference import VonMisesCausalInference#, get_cue_combined_mean_params
import pickle
from scipy.stats import vonmises
from scipy.interpolate import interp1d
from scipy.special import i0, i1
from prev_resp import model_plot, allComparisonHists, serial_dep_plots
import jax.numpy as jnp

def wrap(data,  limits=np.pi/2):
    data = np.where(data < -limits, data + limits*2, data)
    data = np.where(data >= limits, data - limits*2, data)
    return data


def get_vm_samples(mu, kappa, num_samples=1000):
    return vonmises(loc=mu, kappa=kappa).rvs(size=(num_samples))

def kde(data_x: jnp.ndarray, data_y: jnp.ndarray, basis: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """
    Compute the kernel density estimate (KDE) using the von Mises kernel.

    This function calculates the kernel density estimate using the von Mises kernel for two-dimensional data.
    The von Mises kernel is applied separately to the data along the x-axis and y-axis and then combined
    to form the 2D KDE.

    Parameters:
        data_x : jnp.ndarray
            The x-coordinates of the data points represented as a one-dimensional array.
        data_y : jnp.ndarray
            The y-coordinates of the data points represented as a one-dimensional array.
        basis : jnp.ndarray
            The basis for the von Mises kernel, represented as a one-dimensional array.
        kappa : float
            The concentration parameter for the von Mises kernel.

    Returns:
        jnp.ndarray:
            The two-dimensional kernel density estimate (KDE) of the data.

    Notes:
        The von Mises kernel is used to model the circular distribution of the data, and it is based on the
        circular Gaussian distribution.

        The function calculates the kernel density estimate separately along the x-axis and y-axis using
        the von Mises kernel and then combines the results to form the 2D KDE.

        The function assumes that data_x, data_y, and basis have the same number of elements.

        The KDE provides an estimate of the probability density function (PDF) of the underlying distribution
        of the data points in a continuous space.  This function normalises the KDE by the column values
    """
    data_x = jnp.squeeze(data_x)
    data_y = jnp.squeeze(data_y)
    x_ker = jnp.exp(kappa * (jnp.cos(jnp.expand_dims(data_x, 0) - jnp.expand_dims(basis, 1)) - 1))
    y_ker = jnp.exp(kappa * (jnp.cos(jnp.expand_dims(data_y, 0) - jnp.expand_dims(basis, 1)) - 1))
    kde = x_ker @ y_ker.T
    norm = jnp.tile(jnp.reshape(jnp.sum(kde, axis=0), (1, kde.shape[1])), (kde.shape[0], 1))
    return kde / norm


def get_cue_combined_mean_params(mu1, kappa1, mu2, kappa2):
    """
    Determine the mean parameters of the product densities of two von Mises distributions.

    Parameters:
    mu1 (float|ndarray): Mean direction of the first von Mises distribution.
    kappa1 (float|int|ndarray): Concentration parameter of the first von Mises distribution.
    mu2 (float|ndarray): Mean direction of the second von Mises distribution.
    kappa2 (float|int|ndarray): Concentration parameter of the second von Mises distribution.

    Returns:
    tuple: Combined mean direction and concentration parameter.
    """
    mu = wrap(mu2 + np.arctan2(np.sin(mu1-mu2), kappa2/kappa1 + np.cos(mu1-mu2)), limits=np.pi)
    k = np.sqrt((kappa1**2) + (kappa2**2) + 2*kappa1*kappa2*np.cos(mu1 - mu2))
    return mu, k


def get_weighted_cue_combined_mean_params(mu1, kappa1, mu2, kappa2, w1=1.0, w2=1.0):
    """
    Determine the mean parameters of the power-weighted product of two von Mises distributions.

    Parameters:
    mu1 (float or ndarray): Mean direction of the first von Mises.
    kappa1 (float or ndarray): Concentration of the first von Mises.
    mu2 (float or ndarray): Mean direction of the second von Mises.
    kappa2 (float or ndarray): Concentration of the second von Mises.
    w1 (float): Weight/confidence of the first cue (default 1.0).
    w2 (float): Weight/confidence of the second cue (default 1.0).

    Returns:
    tuple: Combined mean direction (mu) and concentration (kappa).
    """
    # Weighted contributions in complex space
    z1 = w1 * kappa1 * np.exp(1j * mu1)
    z2 = w2 * kappa2 * np.exp(1j * mu2)
    z = z1 + z2

    mu = np.angle(z)
    kappa = np.abs(z)
    return mu, kappa


def e_int_to_ext_dist(angle, kappa1=10, sample_size=1000):
    # angle += vonmises(loc=0,kappa=50).rvs(1)
    int_dist = wrap(vonmises(loc=angle, kappa=kappa1).rvs(sample_size))
    ext_dist = unif_map.unif_space_to_angle_space(int_dist)
    return circmean(ext_dist, high=np.pi/2, low=-np.pi/2)


def two_dist_product_out(pdf1, pdf2, theta, theta_scaled, sample_size=1000):
    """
    Combine two distributions into one with product rule
    then wraps the distribution to external space and
    takes the circular mean.
    
    Parameters:
        pdf1, pdf2 : two pdfs to be combined
        theta : domain
        theta_scaled : to 2*pi
    Returns:
        circular mean in external space
    """
    # product of pdfs
    product = pdf1 * pdf2
    product /= np.trapz(product, theta_scaled)  # normalize over original domain

    # plt.plot(theta,pdf1)
    # plt.plot(theta,pdf2)
    # plt.plot(theta,product)
    # plt.show()

    interp_pdf = interp1d(theta, product, kind='linear', bounds_error=False, fill_value=0.0)

    plt.plot(theta,interp_pdf(theta))

    # rejection sampling
    n_samples = sample_size
    samples = []
    max_pdf = np.max(product)

    while len(samples) < n_samples:
        proposal = np.random.uniform(-np.pi/2, np.pi/2)
        u = np.random.uniform(0, max_pdf)
        if u < interp_pdf(proposal):
            samples.append(proposal)

    samples = np.array(samples)

    # plt.hist(samples,bins=100) 
    # plt.show() 

    # s_dist = wrap(vonmises(loc=mu1, kappa=kappa1).rvs(sample_size))
    # r_dist = wrap(vonmises(loc=mu2, kappa=kappa2).rvs(sample_size))
    # breakpoint()
    ext_dist = unif_map.unif_space_to_angle_space(samples)
    # x = kappa1 * np.cos(2 * mu1) + kappa2 * np.cos(2 * mu2)
    # y = kappa1 * np.sin(2 * mu1) + kappa2 * np.sin(2 * mu2)

    return circmean(ext_dist, high=np.pi/2, low=-np.pi/2), product


def combine_von_mises_wrap_out(mu_samp, kappa_samp, kappa1, mu2=0, kappa2=1, pdf2=None, dist=False):
    """
    Combine two von Mises distributions VM(mu1, kappa1) and VM(mu2, kappa2)
    into a single von Mises distribution.
    
    Parameters:
        mu_samp (float): mean of the first von Mises (in radians) from which to sample mean
        kappa_samp (float): concentration of the first von Mises from which to sample mean
        kappa1 (float): concentration of the sampled mean von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises

    Returns:
        circular mean of combined distribution in external space
    """
    # scale angle and mean to standard circle (multiply by 2)
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    theta_scaled = 2 * theta

    # pdfs
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    pdf1 = vonmises.pdf(theta_scaled, kappa1, loc=mu1)
    if dist:
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)
    else:
        pdf2 = vonmises.pdf(theta_scaled, kappa2, loc=2*mu2)
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)


def combine_von_mises_mixture_wrap_out(mu_samp, kappa_samp, kappa1, mu2, kappa2, pdf2=None, dist=False):
    """
    Combine two von Mises distributions VM(mu1, kappa1) and VM(mu2, kappa2)
    into a single von Mises distribution using the reflection mixture of VM2.
    
    Parameters:
        mu_samp (float): mean of the first von Mises (in radians) from which to sample mean
        kappa_samp (float): concentration of the first von Mises from which to sample mean
        kappa1 (float): concentration of the sampled mean von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises

    Returns:
        circular mean of combined distribution in external space
    """
     # scale angle and mean to standard circle (multiply by 2)
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    theta_scaled = 2 * theta

    # pdfs
    # pdfs
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    pdf1 = vonmises.pdf(theta_scaled, kappa1, loc=mu1)
    if dist:
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)
    else:
        pdf2_pos = vonmises.pdf(theta_scaled, loc=2*mu2, kappa=kappa2)
        pdf2_neg = vonmises.pdf(theta_scaled, loc=-2*mu2, kappa=kappa2)
        pdf2 = (pdf2_neg + pdf2_pos)/2.0
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)


def A_of_kappa(k):
    """Mean resultant length A(k) = I1(k)/I0(k)"""
    k = np.asarray(k, dtype=float)
    return i1(k) / i0(k)


def alt_circ_mean_dist(mu1, mu2, kap1, kap2, mix1=0.5, mix2=0.5):
    """
    Calculate circular mean of a mixture of VMs using 1st moment formula
    (Mean is weighted sum of the means of the components weighted by their mean resultant lengths)
    """
    return np.angle(mix1 * A_of_kappa(kap1) * np.exp(1j * mu1) + mix2 * A_of_kappa(kap2) * np.exp(1j * mu2))


def combine_vm_mix_efficient(mu_samp, kappa_samp, kappa1, mu2, kappa2, sample_size=10000, mixture_prob=0.5, 
                             ext_space=True, reject=False, dist=False, cue_comb= False, redist=False, weights=[1,1]):
    """
    NB. This = mixture of products
    Analogue of ``combine_von_mises_mixture_wrap_out``
    
    Sample from the mixture of two product density von Mises distributions derived from VM(mu1, kappa1) 
    and VM(mu2, kappa2), as well as VM(mu1, kappa1) and VM(-mu2, kappa2) respectively, and optionally
    wrap out using a wrapper.

    Now! More efficient :) re-normalised the densities appropriately
    
    Parameters:
        mu_samp (float): mean of the cue von Mises from which sampling will happen (in radians)
        kappa_samp (float): concentration of the sampled von Mises
        kappa1 (float): concentration of the first von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises
        sample_size (int): how many RV samples to draw
        mixture_prob (float between 0 and 1): proportion of the mixture from pdf1
        ext_space (boolean): whether to wrap out and take the external circ mean or compute it internally

    Returns:
        circular mean of combined distribution in external space
    """
    # shortcut VM product density pdfs
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    pdf1_mu, pdf1_kap = get_weighted_cue_combined_mean_params(mu1, kappa1, 2*mu2, kappa2, w1=weights[0], w2=weights[1])
    pdf2_mu, pdf2_kap = get_weighted_cue_combined_mean_params(mu1, kappa1, -2*mu2, kappa2, w1=weights[0], w2=weights[1])
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    theta_scaled = 2 * theta

    if reject == True: # for rejection sampling
        mix =  (vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap))/2
        mix /= np.trapz(mix, theta_scaled)
        interp_pdf = interp1d(theta, mix, kind='linear', bounds_error=False, fill_value=0.0)
        n_samples = sample_size
        samples = []
        max_pdf = np.max((vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap))/2)

        while len(samples) < n_samples:
            proposal = np.random.uniform(-np.pi/2, np.pi/2)
            u = np.random.uniform(0, max_pdf)
            if u < interp_pdf(proposal):
                samples.append(proposal)

        mix_samples = np.array(samples)
    else:
        # sampling
        factor1 = i0(pdf1_kap) #/ (4 * np.pi**2 * i0(kappa1) * i0(kappa2))
        if cue_comb == True:
            factor2 = 0
            factor1 = 1
            # mix_samples = vonmises(loc=pdf1_mu, kappa=pdf1_kap).rvs(int(sample_size)) / 2.0 # normalise sampled rvs to (-pi/2, pi/2]
            # mix_pdf = vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) 
        else:
            factor2 = i0(pdf2_kap) #/ (4 * np.pi**2 * i0(kappa1) * i0(kappa2))
            factor12 = factor1 + factor2
            factor1 /= factor12
            factor2 /= factor12
            
    if ext_space == True: # if the dist is first mapped to external space then the circular mean is taken
        if sfxn == 'card' or sfxn == 'pv': # need to sample from dist itself (no shortcuts? <-- worth double-checking later!)
            try:
                pdf1_samples = vonmises(loc=pdf1_mu, kappa=pdf1_kap).rvs(int(mixture_prob * factor1 * sample_size))
            except ValueError:
                print('too few pdf1')
                pdf1_samples = np.array([])
            try:
                pdf2_samples = vonmises(loc=pdf2_mu, kappa=pdf2_kap).rvs(int((1-mixture_prob) * factor2 * sample_size))
            except ValueError:
                print('too few pdf2')
                pdf2_samples = np.array([])
            mix_samples = np.concatenate((pdf1_samples, pdf2_samples)) / 2.0 # normalise sampled rvs to (-pi, pi]
            ext_dist = unif_map.unif_space_to_angle_space(mix_samples)

            if redist: # if we want to keep the internal response distribution for the next iteration (variable kappa_r)
                kappa_hat, loc_hat, scale_hat = vonmises.fit(mix_samples * 2.0, fscale=1)
                return circmean(ext_dist*2, high=np.pi, low=-np.pi)/2.0, loc_hat, kappa_hat
            else:
                return circmean(ext_dist*2, high=np.pi, low=-np.pi, nan_policy='raise')/2.0 # report/2.0 
        
        elif redist: # if we want to keep the internal response distribution for the next iteration (variable kappa_r)
            try:
                pdf1_samples = vonmises(loc=pdf1_mu, kappa=pdf1_kap).rvs(int(mixture_prob * factor1 * sample_size))
            except ValueError:
                print('too few pdf1')
                pdf1_samples = np.array([])
            try:
                pdf2_samples = vonmises(loc=pdf2_mu, kappa=pdf2_kap).rvs(int((1-mixture_prob) * factor2 * sample_size))
            except ValueError:
                print('too few pdf2')
                pdf2_samples = np.array([])
            mix_samples = np.concatenate((pdf1_samples, pdf2_samples)) / 2.0 # normalise sampled rvs to (-pi, pi]
            kappa_hat, loc_hat, scale_hat = vonmises.fit(mix_samples * 2.0, fscale=1)

            return circmean(ext_dist*2, high=np.pi, low=-np.pi)/2.0, loc_hat, kappa_hat
        
        else: # the fastest route using first moment
            return alt_circ_mean_dist(pdf1_mu, pdf2_mu, pdf1_kap, pdf2_kap, mix1=factor1 * mixture_prob, mix2=factor2 * (1-mixture_prob)) / 2.0

    else: # take arg max in internal space then map out
        mix_pdf = mixture_prob * vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + (1-mixture_prob) * vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap)
        mix_pdf /= np.trapz(mix_pdf, theta_scaled)
        if sfxn == 'card' or sfxn == 'pv':
            return unif_map.unif_space_to_angle_space(theta[np.argmax(mix_pdf)]) #, high=np.pi/2, low=-np.pi/2)
        else:
            return theta[np.argmax(mix_pdf)]
        
def combine_von_mises_mixture_wrap_out(mu_samp, kappa_samp, kappa1, mu2, kappa2, pdf2=None, dist=False):
    """
    Combine two von Mises distributions VM(mu1, kappa1) and VM(mu2, kappa2)
    into a single von Mises distribution using the reflection mixture of VM2.
    
    Parameters:
        mu_samp (float): mean of the first von Mises (in radians) from which to sample mean
        kappa_samp (float): concentration of the first von Mises from which to sample mean
        kappa1 (float): concentration of the sampled mean von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises

    Returns:
        circular mean of combined distribution in external space
    """
     # scale angle and mean to standard circle (multiply by 2)
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    theta_scaled = 2 * theta

    # pdfs
    # pdfs
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    pdf1 = vonmises.pdf(theta_scaled, kappa1, loc=mu1)
    if dist:
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)
    else:
        pdf2_pos = vonmises.pdf(theta_scaled, loc=2*mu2, kappa=kappa2)
        pdf2_neg = vonmises.pdf(theta_scaled, loc=-2*mu2, kappa=kappa2)
        pdf2 = (pdf2_neg + pdf2_pos)/2.0
        return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)


def combine_vm_mix_efficient_background(mu_samp, kappa_samp, kappa1, mu2, kappa2, mu_prev = np.pi/4, sample_size=10000, mixture_prob=0.5, 
                             ext_space=True, reject=False, dist=False, cue_comb= False, redist=False, weights=[1,1]):
    """
    Legacy -- mixture of products with background prior (can be r_{n-2} belief)
    NB. This = mixture of products
    Analogue of ``combine_von_mises_mixture_wrap_out``
    
    Sample from the mixture of two product density von Mises distributions derived from VM(mu1, kappa1) 
    and VM(mu2, kappa2), as well as VM(mu1, kappa1) and VM(-mu2, kappa2) respectively, and optionally
    wrap out using a wrapper.

    Now! More efficient :) re-normalised the densities appropriately
    
    Parameters:
        mu_samp (float): mean of the cue von Mises from which sampling will happen (in radians)
        kappa_samp (float): concentration of the sampled von Mises
        kappa1 (float): concentration of the first von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises
        sample_size (int): how many RV samples to draw
        mixture_prob (float between 0 and 1): proportion of the mixture from pdf1
        ext_space (boolean): whether to wrap out and take the external circ mean or compute it internally

    Returns:
        circular mean of combined distribution in external space
    """
    # shortcut VM product density pdfs
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    mu_bg, kap_bg = get_weighted_cue_combined_mean_params(2*mu2, kappa2, np.abs(mu_prev)*2*np.sign(mu2), kappa2, w1=weights[0], w2=weights[1])
    pdf1_mu, pdf1_kap = get_weighted_cue_combined_mean_params(mu1, kappa1, mu_bg, kappa2, w1=weights[0], w2=weights[1])
    pdf2_mu, pdf2_kap = get_weighted_cue_combined_mean_params(mu1, kappa1, -mu_bg, kappa2, w1=weights[0], w2=weights[1])
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    theta_scaled = 2 * theta

    if reject == True: # for rejection sampling
        mix =  (vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap))/2
        mix /= np.trapz(mix, theta_scaled)
        interp_pdf = interp1d(theta, mix, kind='linear', bounds_error=False, fill_value=0.0)
        n_samples = sample_size
        samples = []
        max_pdf = np.max((vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap))/2)

        while len(samples) < n_samples:
            proposal = np.random.uniform(-np.pi/2, np.pi/2)
            u = np.random.uniform(0, max_pdf)
            if u < interp_pdf(proposal):
                samples.append(proposal)

        mix_samples = np.array(samples)
    else:
        # sampling
        factor1 = i0(pdf1_kap) #/ (4 * np.pi**2 * i0(kappa1) * i0(kappa2))
        if cue_comb == True:
            mix_samples = vonmises(loc=pdf1_mu, kappa=pdf1_kap).rvs(int(sample_size)) / 2.0 # normalise sampled rvs to (-pi/2, pi/2]
            mix_pdf = vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) 
        else:
            factor2 = i0(pdf2_kap) #/ (4 * np.pi**2 * i0(kappa1) * i0(kappa2))
            factor12 = factor1 + factor2
            factor1 /= factor12
            factor2 /= factor12
            try:
                pdf1_samples = vonmises(loc=pdf1_mu, kappa=pdf1_kap).rvs(int(mixture_prob * factor1 * sample_size))
            except ValueError:
                print('too few pdf1')
                pdf1_samples = np.array([])
            try:
                pdf2_samples = vonmises(loc=pdf2_mu, kappa=pdf2_kap).rvs(int((1-mixture_prob) * factor2 * sample_size))
            except ValueError:
                print('too few pdf2')
                pdf2_samples = np.array([])
            mix_samples = np.concatenate((pdf1_samples, pdf2_samples)) / 2.0 # normalise sampled rvs to (-pi, pi]
            mix_pdf = mixture_prob * vonmises.pdf(theta_scaled, loc=pdf1_mu, kappa=pdf1_kap) + (1-mixture_prob) * vonmises.pdf(theta_scaled, loc=pdf2_mu, kappa=pdf2_kap)
            mix_pdf /= np.trapz(mix_pdf, theta_scaled)

    # take circular mean
    # plt.hist(mix_samples, bins=50)
    # plt.show()
    # plt.plot(theta,interp_pdf(theta))
    # plt.show()
    if ext_space == True: # if the dist is first mapped to external space then the circular mean is taken
        if sfxn == 'card' or sfxn == 'pv':
            ext_dist = unif_map.unif_space_to_angle_space(mix_samples)
        else:
            ext_dist = mix_samples

        if redist: # if we want to keep the internal response distribution for the next iteration (variable kappa_r)
            kappa_hat, loc_hat, scale_hat = vonmises.fit(mix_samples * 2.0, fscale=1)
            return circmean(ext_dist*2, high=np.pi, low=-np.pi)/2.0, loc_hat, kappa_hat
        else:
            return circmean(ext_dist*2, high=np.pi, low=-np.pi, nan_policy='raise')/2.0
    else: # take arg max in internal space then map out
        if sfxn == 'card' or sfxn == 'pv':
            return unif_map.unif_space_to_angle_space(theta[np.argmax(mix_pdf)]) #, high=np.pi/2, low=-np.pi/2)
        else:
            return theta[np.argmax(mix_pdf)]
        
def combine_vm_two_stage(mu_samp, kappa_samp, kappa1, mu2, kappa2, mu_prev, kappa_rn2_samp, sample_size=10000, mixture_prob=0.5
                         , ext_space=True, reject=False, dist=False, cue_comb= False, redist=False, weights=[1,1]):
    """
    Combine first rn-2 bimodal belief with rn-1 belief, then combine resultant (bimodal) with stimulus

    Parameters:
    mu_samp (float): mean of the cue von Mises from which sampling will happen (in radians)
    kappa_samp (float): concentration of the sampled von Mises
    kappa1 (float): concentration of the first von Mises
    mu2 (float): mean of the second von Mises (in radians)
    kappa2 (float): concentration of the second von Mises
    sample_size (int): how many RV samples to draw
    mixture_prob (float between 0 and 1): proportion of the mixture from pdf1
    ext_space (boolean): whether to wrap out and take the external circ mean or compute it internally
    Returns:
        circular mean of combined distribution in external space
    """

    mu_bg = combine_vm_mix_efficient(mu_prev, kappa_rn2_samp, kappa2, mu2, kappa2, sample_size=sample_size, mixture_prob=mixture_prob)
    mu_final = combine_vm_mix_efficient(mu_samp, kappa_samp, kappa1, mu_bg, kappa2, sample_size=sample_size,
                                ext_space=ext_space, reject=reject, dist=dist, cue_comb=cue_comb, redist=redist)
    return mu_final


def cue_only(mu_samp, kappa_samp, kappa, ext_space=True, sample_size=10000):
    """
    Forward simulation using stimulus only, i.e. no combination with task auxillary
    (For testing different warping functions)

    Parameters:
        mu_samp (float): mean of the cue von Mises from which sampling will happen (in radians)
        kappa_samp (float): concentration of the sampled von Mises
        kappa (float): concentration of the sensory von Mises (participant inference over the cue belief)
        ext_space (bool): whether to wrap out and take the external circ mean or compute it internally (arg max)
    """
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)
    samples = vonmises(loc=mu1, kappa=kappa).rvs(sample_size)
    # ext_dist = unif_map.unif_space_to_angle_space(samples/2.0)
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    theta_scaled = 2 * theta

    if ext_space == False:
        if sfxn == 'card' or sfxn == 'pv':
            return unif_map.unif_space_to_angle_space(theta[np.argmax(vonmises.pdf(theta_scaled,loc=mu1, kappa=kappa))])
        else:
            return theta[np.argmax((vonmises.pdf(theta_scaled,loc=mu1, kappa=kappa)))]
    else:
        if sfxn == 'card' or sfxn == 'pv':
            ext_dist = unif_map.unif_space_to_angle_space(samples)
        else:
            ext_dist = samples
        return circmean(ext_dist, high=np.pi, low=-np.pi)/2.0



def causal_inf(mu_samp, kappa_samp, kappa1, mu2, kappa2, sign, sample_size=10000, p_common=0.5, 
               reflect_internal=False):
    """
    Perform Bayesian causal inference using the Von Mises distribution.
    This function computes the posterior mean of the causal inference
    given the parameters of the two von Mises distributions.
    Parameters:
        mu_samp (float): mean of the cue von Mises from which sampling will happen (in radians)
        kappa_samp (float): concentration of the sampled von Mises
        kappa1 (float): concentration of the sensory von Mises (participant inference over the cue belief)
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises
    Returns:
        float: posterior mean of the causal inference in radians
    """
    vm_ci = VonMisesCausalInference()
    # vm_ci = GaussianCausalInference()
    mu1 = get_vm_samples(2*mu_samp, kappa_samp, num_samples=1)[0]
    # mu2 = get_vm_samples(2*mu2, kappa2, num_samples=1)
    # if np.abs(mu1) < 0.1:
    #     p_common = 0.1
    if reflect_internal == True:
        sign = np.sign(mu1) * np.sign(mu2)  # reflect the internal response if needed
    
    mu_bg, _, sigma_bg = vm_ci.bayesian_causal_inference(2*mu2, np.pi/2*np.sign(mu2), kappa2, kappa2,
                mu_p=None, sigma_p=None, pi_c = p_common, return_sigma=True)
    
    mu, _, sigma = vm_ci.bayesian_causal_inference(mu1, mu_bg*sign, kappa1, sigma_bg,
                mu_p=None, sigma_p=None, pi_c = p_common, return_sigma=True)
    # print(mu1, mu, sigma)
    if sfxn == 'card' or sfxn == 'pv':
        return unif_map.unif_space_to_angle_space(mu / 2.0), sigma
    else:
        # return wrap(vonmises(loc=vm_ci.bayesian_causal_inference(mu1, 2*mu2*sign, kappa1, kappa2,
        #                                 mu_p=None, sigma_p=None, pi_c = p_common)[0] / 2.0, kappa=25).rvs())
        return wrap(mu / 2.0, limits=np.pi/2), sigma

    
if __name__ == '__main__':
    global sfxn
    sfxn = 'none'

    # gg, ggpart = load_online(name='peter',twowm=True,delay=False)
    # allresp = np.where(np.deg2rad(gg['allResponse']) > np.pi/2, np.deg2rad(gg['allResponse'])-np.pi, np.deg2rad(gg['allResponse']))
    # allresp = np.where(allresp < -np.pi/2, allresp+np.pi, allresp)
    # allstim = np.where(np.deg2rad(gg['allTarget']) > np.pi/2, np.deg2rad(gg['allTarget'])-np.pi, np.deg2rad(gg['allTarget']))
    # allstim = np.where(allstim < -np.pi/2, allstim+np.pi, allstim)
    # alluncu = np.where(np.deg2rad(gg['allOther']) > np.pi/2, np.deg2rad(gg['allOther'])-np.pi, np.deg2rad(gg['allOther']))
    # alluncu = np.where(alluncu < -np.pi/2, alluncu+np.pi, alluncu)


    with open('2wm_exp_good.pkl', 'rb') as f:
        y = pickle.load(f)
    
    allstim = y['stim']
    allresp = y['resp']
    # allComparisonHists(allstim, allresp, allresp[:-1],
    #                 filename='images/reflect_comb/real_data')

    #model_plot(allstim,allresp,binsize=51, plot_lines=True)
    if sfxn == 'pv':
        #with open('data/gam/uniform_model_base_inv_kappa_free.pkl', 'rb') as file:
        with open('data/gam/uniform_base_flexible_means_120_final_fits.pkl', 'rb') as file:
            other_data = pickle.load(file)
        print('PV shaping function -- new one')
        other_data['pdf'] = other_data['mean_pdf'] # for the new unif_map only
        unif_map = usu.UnifMap(data=other_data)
        unif_map.get_cdf_and_inverse_cdf()

        allresp_uni = unif_map.angle_space_to_unif_space(allresp)
        allstim_uni = unif_map.angle_space_to_unif_space(allstim)
        # alluncu_uni = unif_map.angle_space_to_unif_space(alluncu)
    elif sfxn == 'card':
        with open('data/gam/uniform_function_cardinal.obj', 'rb') as f:
            card = pickle.load(f)
        print('cardinal shaping function')
        unif_map = usu.UnifMap(data=card)
        unif_map.get_cdf_and_inverse_cdf()

        allresp_uni = unif_map.angle_space_to_unif_space(allresp)
        allstim_uni = unif_map.angle_space_to_unif_space(allstim)
        # alluncu_uni = unif_map.angle_space_to_unif_space(alluncu)
    else:
        allresp_uni = allresp.copy()
        allstim_uni = allstim.copy()
        # alluncu_uni = alluncu.copy()

    perm = np.random.permutation(len(allstim_uni))
    samps = allstim_uni[perm]
    # uncu = alluncu_uni[perm]
    resps = np.zeros(len(samps))
    
    # kappa_samp = 25
    # kappa_r = 6
    # kappa_s = 10
    # mixpr = 0.5
    # kappa_rn2 = 2

    # kappa_samp = 12.6
    # kappa_r = 11.4
    # kappa_s = 12.6
    # mixpr = 0.5
    # kappa_rn2 = 1.5

    # kappa_samp = 8.2
    # kappa_r = 4.2
    # kappa_s = 8.2
    # mixpr = 0.5
    # kappa_rn2 = 4.2

    # kappa_samp = 19.2
    # kappa_r = 0.13
    # kappa_s = 19.2
    # mixpr = 0.5
    # kappa_rn2 = 0.05
    # #
    # cuecomb = False


    # kappa_samp = 19.2
    # kappa_r = 0.13
    # kappa_s = 19.2
    # mixpr = 0.5
    # kappa_rn2 = 0.05

    # kappa_samp = 8
    # kappa_r = 1.0
    # kappa_s = 8
    # mixpr = 0.5
    # kappa_rn2 = 1.0

    kappa_samp = 25
    kappa_r = 6
    kappa_s = 10
    mixpr = 0.5
    kappa_rn2 = 2

    # initialise
    resps[0] = samps[0]
    wm_object = resps[0]
    sigma = kappa_r
    #resps[1], prod = combine_von_mises_mixture_wrap_out(samps[1], kappa_samp, kappa_s, resps[0], kappa_r)

    for i in tqdm(range(2, len(samps))):
        # noise = np.random.normal(loc=resps[i-1], scale=0.05)
        # resps[i] = cue_only(samps[i], kappa_samp, kappa_s)
        if sfxn == 'pv' or sfxn == 'card':
            wm_object = unif_map.angle_space_to_unif_space(resps[i-1])
        else:
            if np.isnan(resps[i-1]):
                print(resps[i-1],resps[i-2], samps[i-1])
                wm_object = samps[i-1]
                resps[i-1] = samps[i-1]
                breakpoint()
            else: 
                wm_object = resps[i-1]

        # resps[i] = cue_only(samps[i], kappa_samp, kappa_s, ext_space=True)
        resps[i] = combine_vm_two_stage(samps[i], kappa_samp, kappa_s, resps[i-1], kappa_r, resps[i-2], kappa_rn2, mixture_prob=mixpr)
        # resps[i] = combine_vm_mix_efficient(samps[i], kappa_samp, kappa_s, wm_object, kappa_r,sample_size=10000, weights=[1,1],
        #                                                          ext_space=True, mixture_prob = mixpr, cue_comb=cuecomb, redist=False)
        
        # resps[i], sigma = causal_inf(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, np.sign(samps[i])*np.sign(wm_object),
        #                       sample_size=10000, p_common=mixpr, reflect_internal=True)

        # if np.abs(np.sign(samps[i]) - np.sign(wm_object)) == 0:    
        #     resps[i] = causal_inf(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, sample_size=10000, p_common=mixpr)
        # else:
        #     resps[i] = causal_inf(samps[i], kappa_samp, kappa_s, -wm_object, kappa_r, sample_size=10000, p_common=mixpr)
        # # if np.abs(np.sign(samps[i]) - np.sign(wm_object)) == 0:
        #     # resps[i], prod = combine_von_mises_mixture_wrap_out(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, pdf2=prod, dist=True)
        #     #resps[i] = causal_inf(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, sample_size=10000, p_common=mixpr)
        #     resps[i] = combine_vm_mix_efficient(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, sample_size=10000, 
        #                                                             ext_space=True, mixture_prob = mixpr, cue_comb=cuecomb, redist=True)
        # # diff = (np.abs(combine_vm_mix_efficient(samps[i], kappa_s, resps[i-1], kappa_r, sample_size=10000, reject=False)
        # #                -resps[i]))
        # # if diff > 0.05:
        # #     print(diff, samps[i], resps[i])
        # elif np.sign(samps[i]) == 0:
        #     # resps[i], prod = combine_von_mises_mixture_wrap_out(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, pdf2=prod, dist=True)
        #     #resps[i] = causal_inf(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, sample_size=10000, p_common=mixpr)
        #     resps[i] = combine_vm_mix_efficient(samps[i], kappa_samp, kappa_s, wm_object, kappa_r, sample_size=10000, 
        #                                                             ext_space=True, mixture_prob = mixpr, cue_comb=cuecomb, redist=True)
        #     # combine_von_mises_mixture_wrap_out(samps[i], kappa_s, resps[i-1], kappa_r, sample_size=10000)
        # else:
        #     # resps[i], prod = combine_von_mises_mixture_wrap_out(samps[i], kappa_samp, kappa_s, -wm_object, kappa_r, pdf2=prod, dist=True)
        #     #resps[i] = causal_inf(samps[i], kappa_samp, kappa_s, -wm_object, kappa_r, sample_size=10000, p_common=mixpr)
        #     resps[i], wm_object, kappa_r = combine_vm_mix_efficient(samps[i], kappa_samp, kappa_s, -wm_object, kappa_r, sample_size=10000, 
        #                                                             ext_space=True, mixture_prob = mixpr, cue_comb=cuecomb, redist=True)
        #     #combine_von_mises_wrap_out(samps[i], kappa_samp, kappa_s, -wm_object, kappa_r, sample_size=10000, ext_space=True, mixture_prob = 0.5)
        #     # combine_von_mises_mixture_wrap_out(samps[i], kappa_s, -resps[i-1], kappa_r, sample_size=10000)

    # finalresp = np.array([e_int_to_ext_dist(i) for i in allstim_uni])

    
    finalresp = np.array(resps)[perm]
    finalsamp = allstim.copy()[perm]
    # finaluncu = alluncu.copy()

    finalresp = resps.copy()

    # fn = 'resp_noise_vm_cue_comb_reflect_'+sfxn+'_prev_resp_kaps_'+str(kappa_s)+'_kapr_'+str(kappa_r)+'kapsamp'+str(kappa_samp)+'_mixprob_'+str(mixpr)
    fn = 'test_'+sfxn+'_bimodal_rn-2_doubly_'+str(kappa_s)+'kapsamp'+str(kappa_samp)+'_kapr_'+str(kappa_r)+'_mixprob_'+str(mixpr)
    
    # fn = 'true_data_permute'

    # with open('data/forward/'+fn+'.pkl', 'wb') as f:
    #     pickle.dump({'finalresp': finalresp, 'finalsamp': finalsamp, 'finaluncu': allstim}, f)

    model_plot(finalsamp,finalresp,binsize=51, plot_lines=True, filename='figs/forward/'+fn)

    kde2 = kde(finalsamp*2, finalresp*2, jnp.linspace(-jnp.pi,jnp.pi,200), kappa=25)
    plt.imshow(kde2, origin='lower')
    plt.savefig('figs/forward/'+fn+'.png')
    plt.show()

    # density histogram for difference

    # fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    # bins = np.histogram2d(relori, finalresp[1:], bins=binning, density=True)
    # newbins = (bins[0]/np.sum(bins[0],axis=0)).T
    # ax[1,0].imshow(newbins,cmap ='plasma', origin='lower')
    # ax[1,0].set_ylabel(label[2]+' resampled')
    # ax[1,0].set_xlabel(label[1])
    # ax[1,1].margins(y=0)
    # ax[1,1].set_yticks([])
    # ax[1,1].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(newbins.T,axis=1),width=np.pi/(binning-1))
    # ax[0,0].margins(x=0)
    # ax[0,0].grid(False)
    # ax[1,1].grid(False)
    # ax[0,0].axis('off')
    # ax[0,0].set_xticks([])
    # ax[0,1].set_visible(False)
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(newbins.T,axis=0),height=np.pi/(binning-1))
    # # fig.colorbar(bins[3],ax=ax[1,1])
    # plt.show()

    # binning=51
    # fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    # bins = ax[1,0].hist2d(finalsamp, finalresp, bins=binning, cmap ='plasma', density=True)
    # ax[1,0].set_ylabel('$R_n$')
    # ax[1,0].set_xlabel('$S_n$')
    # ax[1,1].margins(y=0)
    # ax[1,1].set_yticks([])
    # ax[1,1].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    # ax[0,0].margins(x=0)
    # ax[0,0].grid(False)
    # ax[1,1].grid(False)
    # ax[0,0].axis('off')
    # ax[0,0].set_xticks([])
    # ax[0,1].set_visible(False)
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    # fig.colorbar(bins[3],ax=ax[1,1])
    # plt.savefig('images/reflect_comb/'+fn+'histo_.png', dpi=150)
    # plt.show()

    # allComparisonHists(finalsamp, finalresp, finaluncu[1:],filename=fn)
    allComparisonHists((finalsamp), (finalresp), (finalresp[:-1]),
                       filename='figs/forward/'+fn)
    
    serial_dep_plots(finalsamp, finalresp, allstim, allresp, fn=fn)
    
    #allComparisonHists_norm(finalsamp, finalresp, finalresp[:-1])
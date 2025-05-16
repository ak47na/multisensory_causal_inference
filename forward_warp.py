import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d, circmean
from load import load_online
import unif_space_utils as usu
from tqdm import tqdm
from prev_resp import model_plot
from von_mises_causal_inf import VonMisesCausalInference
import pickle
from scipy.stats import vonmises
from scipy.interpolate import interp1d

def wrap(data):
    data = np.where(data < -np.pi/2, data + np.pi, data)
    data = np.where(data >= np.pi/2, data - np.pi, data)
    return data

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

    # rejection sampling
    n_samples = 1000
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

    return circmean(ext_dist, high=np.pi/2, low=-np.pi/2)

def combine_von_mises_wrap_out(mu1, kappa1, mu2, kappa2):
    """
    Combine two von Mises distributions VM(mu1, kappa1) and VM(mu2, kappa2)
    into a single von Mises distribution.
    
    Parameters:
        mu1 (float): mean of the first von Mises (in radians)
        kappa1 (float): concentration of the first von Mises
        mu2 (float): mean of the second von Mises (in radians)
        kappa2 (float): concentration of the second von Mises

    Returns:
        circular mean of combined distribution in external space
    """
    # scale angle and mean to standard circle (multiply by 2)
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    theta_scaled = 2 * theta

    # pdfs
    pdf1 = vonmises.pdf(theta_scaled, kappa1, loc=2*mu1)
    pdf2 = vonmises.pdf(theta_scaled, kappa2, loc=2*mu2)

    return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)


def combine_von_mises_mixture_wrap_out(mu1, kappa1, mu2, kappa2, sample_size=1000):
     # scale angle and mean to standard circle (multiply by 2)
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    theta_scaled = 2 * theta

    # pdfs
    pdf1 = vonmises.pdf(theta_scaled, kappa1, loc=2*mu1)
    pdf2_pos = vonmises.pdf(theta_scaled, kappa2, loc=2*mu2)
    pdf2_neg = vonmises.pdf(theta_scaled, kappa2, loc=-2*mu2)
    pdf2 = (pdf2_neg + pdf2_pos)/2.0

    return two_dist_product_out(pdf1, pdf2, theta, theta_scaled)

gg, ggpart = load_online(name='peter',twowm=True,delay=False)
allresp = np.where(np.deg2rad(gg['allResponse']) > np.pi/2, np.deg2rad(gg['allResponse'])-np.pi, np.deg2rad(gg['allResponse']))
allresp = np.where(allresp < -np.pi/2, allresp+np.pi, allresp)
allstim = np.where(np.deg2rad(gg['allTarget']) > np.pi/2, np.deg2rad(gg['allTarget'])-np.pi, np.deg2rad(gg['allTarget']))
allstim = np.where(allstim < -np.pi/2, allstim+np.pi, allstim)

with open('data/gam/uniform_model_base_inv_kappa_free.pkl', 'rb') as file:
    other_data = pickle.load(file)

unif_map = usu.UnifMap(data=other_data)
unif_map.get_cdf_and_inverse_cdf()

allresp_uni = unif_map.angle_space_to_unif_space(allresp)
allstim_uni = unif_map.angle_space_to_unif_space(allstim)

samps = np.random.permutation(allstim_uni)[:5000]
resps = np.zeros(len(samps))
resps[0] = samps[0]

kappa_r = 10
kappa_s = 20

plt.hist(samps,bins=100)
plt.show()

for i in tqdm(range(1, len(samps))):
    # noise = np.random.normal(loc=resps[i-1], scale=0.05)
    if np.abs(np.sign(resps[i-1]) - np.sign(samps[i])) == 0:
        resps[i] = combine_von_mises_mixture_wrap_out(samps[i], kappa_s, resps[i-1], kappa_r)
    elif np.sign(samps[i]) == 0:
        resps[i] = combine_von_mises_mixture_wrap_out(samps[i], kappa_s, samps[i], kappa_r)
    else:
        resps[i] = combine_von_mises_mixture_wrap_out(samps[i], kappa_s, -resps[i-1], kappa_r)

# finalresp = np.array([e_int_to_ext_dist(i) for i in allstim_uni])

finalresp = np.array(resps)
finalsamp = unif_map.unif_space_to_angle_space(samps)

binning=41
fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
bins = ax[1,0].hist2d(finalsamp, finalresp, bins=binning, cmap ='plasma', density=True)
ax[1,0].set_ylabel('$R_n$')
ax[1,0].set_xlabel('$S_n$')
ax[1,1].margins(y=0)
ax[1,1].set_yticks([])
ax[1,1].axis('off')
ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
ax[0,0].margins(x=0)
ax[0,0].grid(False)
ax[1,1].grid(False)
ax[0,0].axis('off')
ax[0,0].set_xticks([])
ax[0,1].set_visible(False)
ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
fig.colorbar(bins[3],ax=ax[1,1])
plt.show()
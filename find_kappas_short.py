import numpy as np
from scipy.special import i0, i1, i0e
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


# ------------------------ utilities ------------------------
def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi


# -------------------- jax utilities ------------------------
def wrap_jax(a, division=1.0):
    return (a + jnp.pi/division) % (2*jnp.pi/division) - jnp.pi/division

def log_vm_jax(theta, mu, kappa, div=1.0):
    return kappa*jnp.cos(wrap_jax(theta - mu, division=div) * div) - jnp.log(jnp.i0(kappa) + 1e-12) - jnp.log(2*jnp.pi)


# ------------------------ model fit ------------------------

# --- helpers ---

def x_infer_jax(ka, kx, a, r, division = 1.0):
    """
    geometric solution for x given a, r, kx, ka
    - ensure ka < kx to keep arcsin domain
    - clip arcsin argument to (-1+eps, 1-eps)
    """
    ka = jnp.minimum(ka, kx - 1e-6)

    # case 1: sign(a) == sign(r)
    arg1 = (ka/kx) * jnp.sin(jnp.abs(r*division) - jnp.abs(a*division))
    arg1 = jnp.clip(arg1, -1.0 + 1e-6, 1.0 - 1e-6)
    x1 = jnp.arcsin(arg1) + jnp.abs(r*division)

    # case 2: sign(a) != sign(r)
    arg2 = (ka/kx) * jnp.sin(jnp.abs(a*division) + jnp.abs(r*division))
    arg2 = jnp.clip(arg2, -1.0 + 1e-6, 1.0 - 1e-6)
    x2 = jnp.arcsin(arg2) + jnp.abs(r*division)

    return jnp.where(jnp.sign(a) == jnp.sign(r), x1 / division, x2 / division)

def cosyne_rule_jax(k1, k2, a, x, division = 1.0):
    value = jnp.sqrt(k1**2 + k2**2 - 2*k1*k2*jnp.cos(jnp.pi - division * (a + x)))
    return jnp.where(jnp.isnan(value), 1e-6, value)

# --- main fitting function ---

def param_fit_vector(resp, stim, ks0, kr1, kr2, division=1.0, m=0.5, steps=1000, lr=0.01, tol=1e-6, plotting=False):
    """
    Fit parameters ks, kx, kr1, kr2 using NLL loss minimisation
    resp, stim: in radians
    ks0, kx0, kr1, kr2: initial parameters for concentrations

    Parameters:
    division: factor to divide angles by for fitting (1.0 for full circle, 2.0 for half circle)
    m: mixture weight for previous response (0.5 for equal weight) -- not activated
    steps: number of gradient descent steps
    lr: learning rate
    tol: tolerance for gradient norm to stop optimization
    plotting: whether to plot parameter trace and fits

    Returns:
    dict with fitted parameters ks0, kx0, kr1, kr2
    """
    # initialisation
    theta = jnp.array([ks0, kr1, kr2], dtype=jnp.float32)
    N = len(resp)
    resp = jnp.asarray(resp)
    stim = jnp.asarray(stim)
    stim_sign = jnp.sign(stim)
    resp_sign = jnp.sign(resp)
    resp_prev = jnp.roll(resp, 1)
    resp_prev = resp_prev.at[0].set(stim[0])
    resp_prev_2 = jnp.roll(resp_prev, 1)
    resp_prev_2 = resp_prev_2.at[0:2].set(stim[0:2])
    resp_prev_sign = jnp.sign(resp_prev) * jnp.sign(stim)
    resp_prev_2_sign = jnp.sign(resp_prev_2) * jnp.sign(stim)
    resp_prev_signed = resp_prev * resp_prev_sign
    resp_prev_2_signed = resp_prev_2 * resp_prev_2_sign

    # use absolute values for fitting (everthing in positive quadrant)
    resp_abs = jnp.abs(resp)
    resp_prev_abs = jnp.abs(resp_prev)
    resp_prev_2_abs = jnp.abs(resp_prev_2)

    if stim.max() < jnp.pi/(division+1) + 0.01: # wrap angles if not done already
        print('wrapping angles to (-pi/{}, pi/{})'.format(division, division))
        resp = wrap(2.0 * resp)
        stim = wrap(2.0 * stim)
        resp_prev_2 = wrap(2.0 * resp_prev_2)
        resp_prev = wrap(2.0 * resp_prev)

    # unit vectors relative to stim (tried; removed)
    # resp = wrap(resp - stim)
    # resp_prev = wrap(resp_prev - stim)
    # resp_prev_2 = wrap(resp_prev_2 - stim)
    
    division2 = 1.0 # keep actual angle value for trigonometric functions
    # unit vectors of +- relative vectors
    y = jnp.stack([jnp.cos(division2 * resp_abs), jnp.sin(division2 * resp_abs)], axis=-1)
    # y = jnp.exp(1j * resp)
    vec_s = jnp.stack([jnp.cos(division2 * stim), jnp.sin(division2 * stim)], axis=-1)
    vec_res_p = jnp.stack([jnp.cos(division2 * resp_prev_abs), jnp.sin(division2 * resp_prev_abs)], axis=-1)
    # vec_res_p = jnp.exp(1j * resp_prev)
    vec_res_m = jnp.stack([jnp.cos(division2 * -resp_prev_abs), jnp.sin(division2 * -resp_prev_abs)], axis=-1)
    # vec_res_m = jnp.exp(1j * (- resp_prev))
    vec_res_2_p = jnp.stack([jnp.cos(division2 * resp_prev_2_abs), jnp.sin(division2 * resp_prev_2_abs)], axis=-1)
    vec_res_2_m = jnp.stack([jnp.cos(division2 * -resp_prev_2_abs), jnp.sin(division2 * -resp_prev_2_abs)], axis=-1)
    # vec_res_2_p = jnp.exp(1j * resp_prev_2)
    # vec_res_2_m = jnp.exp(1j * (-resp_prev_2))
    
    # NLL loss
    def loss_vec(th, ret_x = False, division = division):
        k_s = th[0]
        k_r = th[1]
        k_r2 = th[2]

        # combine to calculate auxiliary vector
        aux_p = k_r * vec_res_p + k_r2 * vec_res_2_p
        # aux_m = k_r * vec_res_m + k_r2 * vec_res_2_m # we don't use this currently
        k_aux_p = jnp.linalg.norm(aux_p, axis=-1) + 1e-12
        # k_aux_m = jnp.linalg.norm(aux_m, axis=-1) + 1e-12
        unit_vec_aux_p = aux_p / k_aux_p[:, None]
        # unit_vec_aux_m = aux_m / k_aux_m[:, None]

        aux_p_ang = wrap_jax(jnp.arctan2(unit_vec_aux_p[:,1], unit_vec_aux_p[:,0]) / division2,
                              division=division)
        aux_p_wrap = k_aux_p[: , None] * jnp.stack([jnp.cos(aux_p_ang * division2),
                                                     jnp.sin(aux_p_ang * division2)], axis=-1)
        # aux_m_ang = wrap_jax(jnp.arctan2(unit_vec_aux_m[:,1], unit_vec_aux_m[:,0]) / division2, division=division)

        if jnp.any(jnp.isnan(unit_vec_aux_p)): # or jnp.any(jnp.isnan(x_m)):
            print(th)
            print('nan ang')
            breakpoint()
        
        # calculate x (positive only)
        x_p = wrap_jax(x_infer_jax(k_aux_p, k_s, aux_p_ang, resp_abs, division=division2), division=division)

        if jnp.any(jnp.isnan(x_p)): # or jnp.any(jnp.isnan(x_m)):
            print(th)
            print('nan x')
            breakpoint()

        vec_x_p = jnp.stack([jnp.cos(division2 * x_p), jnp.sin(division2 * x_p)], axis=-1)

        if jnp.any(jnp.isnan(vec_x_p)): # or jnp.any(jnp.isnan(x_m)):
            print(th)
            print('nan x')
            breakpoint()

        # calculate combined kappa for r / prediction if desired
        k_tot = cosyne_rule_jax(k_aux_p, k_s, aux_p_ang, x_p, division = division2) + 1e-12

        if jnp.any(jnp.isnan(k_tot)): # or jnp.any(jnp.isnan(x_m)):
            print(th)
            print('nan k_tot')
            breakpoint()

        # prediction (we don't use this for loss in current version so could return just when asked)
        # diff_x = vec_x_p - vec_s
        pred = k_s * vec_x_p - aux_p_wrap
        k_pred = jnp.linalg.norm(pred, axis=-1) + 1e-12
        unit_pred = pred / k_pred[:, None]
        pred_angle = wrap_jax(jnp.arctan2(unit_pred[:,1] * division2, unit_pred[:,0] * division2) / division2, 
                              division=division)

        # loss (negative log-likelihood) -- currently including likelihood both for x and r (could use just x or r)
        # loss = k_tot[:, None] * y - k_pred[:, None] * jnp.stack([jnp.cos(pred_angle), jnp.sin(pred_angle)], axis=-1) ## old -- vector loss
        nll = - log_vm_jax(x_p, stim_sign * stim, k_s, div=division) - log_vm_jax(resp_abs, pred_angle, k_tot, div=division) 

        if ret_x:
            return stim, stim_sign*x_p, resp, resp_sign*pred_angle
        return jnp.sum(nll) #old versions -- vector loss: jnp.mean(jnp.sum(loss * loss, axis=-1)) #+ jnp.mean(jnp.sum(diff_x * diff_x, axis = -1)) 
    
    # grad descent
    loss_grad = grad(lambda th: loss_vec(th))
    losses = np.zeros((4, steps//10), dtype = float)
    
    for _ in tqdm(range(steps)):
        g = loss_grad(theta)

        if jnp.any(jnp.isnan(g)):
            print('nan grad')
            print(theta)
            g = jnp.where(jnp.isnan(g), 1e-3, g)
            breakpoint()
        # print(g)
        if jnp.abs(jnp.sum(g)) < tol:
            print('achieved',g)
            break
        
        # update parameters
        theta = theta - lr * g
        theta = jnp.where(theta < 0, 1.0, theta) # reset if negative

        if _ % 10 == 0:
            losses[0,_//10] = loss_vec(theta)
            losses[1:,_//10] = theta
            print(losses[:, _//10])
    
    # optional plotting
    if plotting:
        plt.plot(losses[1,:], label='k_s param')
        plt.plot(losses[2,:], label='k_r1 param')
        plt.plot(losses[3,:], label='k_r2 param')
        plt.plot(losses[0, :], label= 'loss', linestyle='dashed', color='black')
        plt.legend()
        plt.title('Parameter optimization trace')
        plt.xlabel('Step')
        plt.ylabel('Parameter value')
        plt.yscale('log')
        plt.show()

        s_x_r_y = loss_vec(theta, ret_x=True)

        plt.hist2d(s_x_r_y[0], s_x_r_y[1],bins=50)
        plt.xlabel('stimulus')
        plt.ylabel('inferred x')
        plt.title('Stimulus vs inferred x')
        plt.show()

        plt.hist2d(s_x_r_y[2], s_x_r_y[3], bins=50)
        plt.xlabel('response')
        plt.ylabel('predicted response')
        plt.title('Response vs predicted response')
        plt.show()

        plt.hist2d(s_x_r_y[1], s_x_r_y[3], bins=50)
        plt.xlabel('inferred x')
        plt.ylabel('predicted response')
        plt.title('Inferred x vs predicted response')
        plt.show()

        plt.hist2d(s_x_r_y[1], s_x_r_y[2], bins=50)
        plt.xlabel('inferred x')
        plt.ylabel('response')
        plt.title('Inferred x vs true response')
        plt.show()

    return {'k_s': theta[0], 'k_r': theta[1], 'k_r2' : theta[2]}


if __name__ == "__main__":
    # replace loading by whichever dataset procedure (e.g. pickled files)
    with open('2wm_exp_good.pkl', 'rb') as f:
        y = pickle.load(f)

    # only use trials where response and stimulus have same sign (to avoid confusion from mistrials)
    allresp2 = y['resp'][np.argwhere(y['resp']*y['stim'] > 0)][:,0]
    allstim2 = y['stim'][np.argwhere(y['resp']*y['stim'] > 0)][:,0]


    out = param_fit_vector(allresp2, allstim2, 20, 10, 5, lr=0.001, steps=200, division=2.0, plotting=True)
    print(out)

    # ideally should test found params in forward_warp.py to see if they match forward simulation and simulation metrics well
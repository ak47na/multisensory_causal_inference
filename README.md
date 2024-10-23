# README

## Causal Inference in Multisensory Perception

This repository implements the models described in the paper "Causal Inference in the Multisensory Brain." The project includes the derivation and implementation of segregated, fused, and causal inference estimates for multisensory perception.

### Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Models](#models)
    - [Segregated Estimate](#segregated-estimate)
    - [Fused Estimate](#fused-estimate)
    - [Causal Inference Estimate](#causal-inference-estimate)
4. [Our Causal Inference Forward Model](#our-model)
5. [References](#references)

## Introduction
Multisensory perception involves integrating information from different sensory modalities (e.g., visual and auditory). This project implements three models for estimating the true stimulus rate based on sensory inputs: segregated, fused, and causal inference estimates.

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/ak47na/multisensory_causal_inference.git
cd multisensory_causal_inference
```

## Models

### Segregated Estimate
In the segregation case, the sensory inputs are processed independently. The segregated estimate for each modality is given by:
$\hat{s}_{v, c=2} = \frac{x_v \sigma_p^2 + \mu_p \sigma_v^2}{\sigma_v^2 + \sigma_p^2}$

$\hat{s}_{a, c=2} = \frac{x_a \sigma_p^2 + \mu_p \sigma_a^2}{\sigma_a^2 + \sigma_p^2}$

### Fused Estimate
In the fusion case, the sensory inputs are combined based on their relative reliabilities. The fused estimate is:

$\hat{s} = \frac{x_v \sigma_a^2 \sigma_p^2 + x_a \sigma_v^2 \sigma_p^2 + \mu_p \sigma_v^2 \sigma_a^2}{\sigma_v^2 \sigma_a^2 + \sigma_v^2 \sigma_p^2 + \sigma_a^2 \sigma_p^2}$

### Causal Inference Estimate
The causal inference model incorporates the belief about whether the sensory inputs come from a common source or separate sources. The estimate is:

$\hat{s}_v = p(c = 1 | x_v, x_a) \hat{s}_{v, c=1} + p(c = 2 | x_v, x_a) \hat{s}_{v, c=2}$

$\hat{s}_a = p(c = 1 | x_v, x_a) \hat{s}_{a, c=1} + p(c = 2 | x_v, x_a) \hat{s}_{a, c=2}$

Where the posterior probability of the common cause hypothesis is:

$p(c = 1 | x_v, x_a) = \frac{p(x_v, x_a | c = 1) \pi_c}{p(x_v, x_a | c = 1) \pi_c + p(x_v, x_a | c = 2) (1 - \pi_c)}$

## Our Causal Inference Forward Model

For a tutorial of the model, please see `example_custom_causal_inference.ipynb`.

We estimate angle responses by performing causal inference in an internal space where angles are uniformly distributed, converting using the mapping function $U$. The optimal estimates are determined by mapping the posterior distributons back to the angles space using $U^{-1}$. 

Concretely, consider $t, s_n$ to be the means of the two cues. We select a grid of pairs $(t, s_n)$ with $s_n$ constant across columns and $t$ constant across rows and convert the values to internal space resulting in pairs $(ut, us_n)$. 

For each grid cell we sample `num_sim` internal samples $x_1 \sim VM(ut, \kappa_1), x_2 \sim VM(us_n, \kappa_2)$ using `get_vm_samples` inside `forward_models_causal_inference.py`. 
For each pair of samples $x_1, x_2$ we:
1. compute the posteriors $p(u|x_1, x_2, C=1) \propto p(x_1, x_2|u)p(u)$ and $p(ut|x_1, C=2) \propto p(x_1|ut)p(ut), p(us_n|x_2, C=2) \propto p(x_2|us_n)p(us_n)$ which are Von Mises
2. convert the Von Mises posteriors from internal space to angle space using $U^{-1}$ and select the circular mean of the distribution as the optimal estimates $\hat{s}_n, \hat{t}$ for $C=1$ using `fusion_estimate` and for $C=2$ using `segregation_estimate` inside `custom_causal_inference.py`
3. compute the posteriors probability of common cause in internal space (using `posterior_prob_common_cause` inherited from ` VonMisesCausalInference`) and use it to combine $\hat{s}^{C=1}_n = \hat{t}^{C=1}, \hat{s}^{C=2}_n, \hat{t}^{C=2}$ into the final optimal estimates (`bayesian_causal_inference` in `custom_causal_inference.py`).

Finally, we take the circular mean across sample pairs $x_1, x_2$ to obtain the "mean" estimates of $s_n$ as a function of $t, s_n$ for all $(t, s_n)$ grid pairs.


## References
- Ernst, M. O., & Bülthoff, H. H. (2004). Merging the senses into a robust percept. Trends in Cognitive Sciences, 8(4), 162-169.
- Kording, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007). Causal inference in multisensory perception. PLOS ONE, 2(9), e943.
- Wozny, D. R., Beierholm, U., & Shams, L. (2010). Probability matching as a computational strategy used in perception. PLOS Computational Biology, 6(8), e1000871.

---

For further information, please refer to the original paper: Causal inference in multisensory perception.
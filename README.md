# README

## Causal Inference in Multisensory Perception

This repository implements the models described in the paper "Causal Inference in the Multisensory Brain." The project includes the derivation and implementation of segregated, fused, and causal inference estimates for multisensory perception.

### Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Models](#models)
    - [Segregated Estimate](#segregated-estimate)
    - [Fused Estimate](#fused-estimate)
    - [Causal Inference Estimate](#causal-inference-estimate)
5. [Examples](#examples)
6. [References](#references)

## Introduction
Multisensory perception involves integrating information from different sensory modalities (e.g., visual and auditory). This project implements three models for estimating the true stimulus rate based on sensory inputs: segregated, fused, and causal inference estimates.

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/ak47na/causal_inference.git
cd causal_inference
```

## Models

### Segregated Estimate
In the segregation case, the sensory inputs are processed independently. The segregated estimate for each modality is given by:
\[
\hat{s_v} = \frac{r_v \sigma_p^2 + \mu_p \sigma_v^2}{\sigma_v^2 + \sigma_p^2}
\]
\[
\hat{s_a} = \frac{r_a \sigma_p^2 + \mu_p \sigma_a^2}{\sigma_a^2 + \sigma_p^2}
\]

### Fused Estimate
In the fusion case, the sensory inputs are combined based on their relative reliabilities. The fused estimate is:
\[
\hat{s} = \frac{r_v \sigma_a^2 \sigma_p^2 + r_a \sigma_v^2 \sigma_p^2 + \mu_p \sigma_v^2 \sigma_a^2}{\sigma_v^2 \sigma_a^2 + \sigma_v^2 \sigma_p^2 + \sigma_a^2 \sigma_p^2}
\]

### Causal Inference Estimate
The causal inference model incorporates the belief about whether the sensory inputs come from a common source or separate sources. The estimate is:
\[
\hat{s} = p(c = 1 | r_v, r_a) \hat{s}_{\text{fusion}} + p(c = 2 | r_v, r_a) \hat{s}_{\text{segregation}}
\]
Where the posterior probability of the common cause hypothesis is:
\[
p(c = 1 | r_v, r_a) = \frac{p(r_v, r_a | c = 1) \pi_c}{p(r_v, r_a | c = 1) \pi_c + p(r_v, r_a | c = 2) (1 - \pi_c)}
\]

## Examples
Check the `examples` folder for detailed usage examples and scripts demonstrating the application of the models.

## References
- Ernst, M. O., & Bülthoff, H. H. (2004). Merging the senses into a robust percept. Trends in Cognitive Sciences, 8(4), 162-169.
- Kording, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007). Causal inference in multisensory perception. PLOS ONE, 2(9), e943.
- Wozny, D. R., Beierholm, U., & Shams, L. (2010). Probability matching as a computational strategy used in perception. PLOS Computational Biology, 6(8), e1000871.

---

For further information, please refer to the original paper: Causal inference in multisensory perception.
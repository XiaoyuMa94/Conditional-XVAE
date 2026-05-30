# Conditional-XVAE (cXVAE) Tutorial

Modeling Spatial-Temporal Extremes via Conditional Variational Autoencoders 

> Department of Statistics and Data Science, University of Missouri Columbia

## Introduction

This tutorial provides step-by-step instructions for implementing the cXVAE model, which extends the XVAE framework [[1]](#1) to incorporate climate indices as conditioning variables. By allowing the latent extremal-dependence parameters to evolve with observed climate drivers (e.g., ENSO), the model moves beyond stationarity assumptions and enables controlled generation of spatial extreme fields under specified climate conditions.

The implementation is in Python using PyTorch, with data preparation scripts in R. This work builds on the XVAE framework of Zhang et al. (2026) [[1]](#1) and is described in Ma et al. (2025) [[2]](#2).

## Dependencies

**R packages**
```r
install.packages(c("fields", "mgcv", "xts", "ismev", "ggplot2", "parallel", "zoo"))
```

**Python packages**
```bash
pip install torch numpy pandas scipy matplotlib cartopy tqdm imageio pillow
```

All experiments in the paper were performed on a desktop with an Intel Core i5-9600K CPU (3.70 GHz) and 48 GB RAM. No GPU is required.

## Repository Structure
 
```
Conditional-XVAE/
├── Simulation/
│   ├── simulate_data.R          # Data generation from the max-id model
│   ├── pretrain_CNN.py          # CNN pretraining for theta estimation
│   ├── train_cXVAE.py           # cXVAE model definition and training
│   ├── baseline_SohnCVAE.py     # Vanilla cVAE baseline (Sohn et al., 2015)
│   ├── evaluate.py              # Evaluation metrics and visualizations
│   └── utils.py                 # Shared utility functions
│
├── FWI Data Analysis/
│   ├── data_preparation.R       # FWI data processing and GEV transformation
│   ├── pretrain_CNN_FWI.py      # CNN pretraining for FWI analysis
│   ├── train_cXVAE_FWI.py       # cXVAE training and evaluation for FWI
│   └── utils.py                 # Shared utility functions
│
├── figures/                     # Result figures
├── README.md
└── LICENSE
```
 
---


## References
 
<a id="1">[1]</a>
Zhang, L., Ma, X., Wikle, C.K. and Huser, R. "Fast and flexible emulation of spatial extremes processes via variational autoencoders." *Journal of the American Statistical Association*, just-accepted, 2026.
 
<a id="2">[2]</a>
Ma, X., Zhang, L. and Wikle, C.K. "Modeling spatio-temporal extremes via conditional variational autoencoders." *Annals of Applied Statistics*, under review, 2025.
 
<a id="3">[3]</a>
Sohn, K., Lee, H. and Yan, X. "Learning structured output representation using deep conditional generative models." *Advances in Neural Information Processing Systems*, 28, 2015.


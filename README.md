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

## Part I: Simulation Study
 
The simulation study demonstrates the cXVAE on a 50×50 regular grid over $[0,20]\times[0,20]$, conditioned on the ENSO index (528 monthly time points, 1980–2023).

### Step 1: Generate Simulation Data (R)

Download the ENSO index from NOAA:
[https://psl.noaa.gov/data/climateindices/list/](https://psl.noaa.gov/data/climateindices/list/)
 
Then run the simulation script:
 
```r
source("Simulation/simulate_data.R")
```

The ENSO time series used as the condition variable $c_t$ is shown below. The three red dashed lines mark the representative El Niño, neutral, and La Niña periods used in the paper.

![ENSO time series](figures/ENSO.png)

The tilting parameter field $\theta_t(c_t)$ governs the spatial extent of extremal dependence and evolves with the ENSO index. Low values of $\theta$ correspond to heavier-tailed latent variables. Below are three representative $\theta_t$ maps at El Niño (left), neutral (center), and La Niña (right) conditions:

| El Niño | Neutral | La Niña |
|:---:|:---:|:---:|
| ![theta El Nino](figures/theta1.png) | ![theta Neutral](figures/theta2.png) | ![theta La Nina](figures/theta3.png) |

**Outputs** (saved as CSV):
 
| File | Description | Shape |
|------|-------------|-------|
| `X_Data.csv` | Observed spatial fields | 2500 × 528 |
| `Y_Data.csv` | Latent Y process | 2500 × 528 |
| `Z_Data.csv` | Latent expPS variables | 225 × 528 |
| `Thetas_Data.csv` | Tilting parameters | 225 × 528 |
| `W_Data.csv` | Wendland basis matrix | 2500 × 225 |
| `RBF_Data.csv` | RBF basis for theta | 225 × 144 |
| `MEIs_MA_Data.csv` | Smoothed ENSO index | 528 × 1 |

### Step 2: Pretrain the CNN (Python)
 
The CNN decoder is pretrained to estimate the tilting parameter field $\theta_t$ from the approximation of fused latent inputs $(\hat{Z}_t, c_t)$. This warm-start is strongly recommended before training the full cXVAE.
 
```python
python Simulation/pretrain_CNN.py
```
 
The pretrained weights are saved to `CNN_pretrained.pt`.


 
## References
 
<a id="1">[1]</a>
Zhang, L., Ma, X., Wikle, C.K. and Huser, R. "Fast and flexible emulation of spatial extremes processes via variational autoencoders." *Journal of the American Statistical Association*, just-accepted, 2026.
 
<a id="2">[2]</a>
Ma, X., Zhang, L. and Wikle, C.K. "Modeling spatio-temporal extremes via conditional variational autoencoders." *Annals of Applied Statistics*, under review, 2025.
 
<a id="3">[3]</a>
Sohn, K., Lee, H. and Yan, X. "Learning structured output representation using deep conditional generative models." *Advances in Neural Information Processing Systems*, 28, 2015.


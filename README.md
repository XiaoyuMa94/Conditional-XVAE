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

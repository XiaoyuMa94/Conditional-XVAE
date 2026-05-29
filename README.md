# Conditional-XVAE (cXVAE) Tutorial

Modeling Spatial-Temporal Extremes via Conditional Variational Autoencoders 

> Department of Statistics and Data Science, University of Missouri Columbia

## Introduction

This tutorial provides step-by-step instructions for implementing the cXVAE model, which extends the XVAE framework to incorporate climate indices as conditioning variables. The model allows extremal dependence parameters to vary over time with observed climate drivers (e.g., ENSO), enabling counterfactual experiments and non-stationary emulation of spatial extreme fields. The implementation is in Python using PyTorch.

This work builds on the XVAE framework of Zhang et al. (2026) [[1]](#1) and CVAE framework of Sohn et al. (2015) [[2]](#2), and described in Ma et al. (2025) [[3]](#3).

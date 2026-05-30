# =============================================================================
# utils.py
#
# Utility functions for the cXVAE project:
#   - Data augmentation (pseudo-replicate construction)
#   - Spatial basis functions (Wendland, RBF)
#   - Custom PyTorch Dataset classes
#   - Evaluation metrics: chi-coefficient, ARE
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# Numerical Utilities
# =============================================================================

def softplus_clip(x, beta):
    """Numerically stable softplus with clipping to prevent overflow."""
    x = x.clone()
    x = torch.where(x * beta > 700, torch.tensor(700.0 / beta), x)
    return (1 / beta) * torch.log1p(torch.exp(beta * x))


# =============================================================================
# Data Augmentation: Pseudo-Replicate Construction
# =============================================================================

def x_aug(x):
    """
    Augment a (K x n_t) tensor into (n_t x 3K) by stacking
    [t-1, t, t+1] time slices for each time point.

    """
    x_tmp = torch.zeros(x.shape[1], x.shape[0] * 3)
    for t in range(x.shape[1]):
        if t == 0:
            x_tmp[t, :] = torch.cat([x[:, t], x[:, t], x[:, t + 1]])
        elif t == (x.shape[1] - 1):
            x_tmp[t, :] = torch.cat([x[:, t - 1], x[:, t], x[:, t]])
        else:
            x_tmp[t, :] = torch.cat([x[:, t - 1], x[:, t], x[:, t + 1]])
    return x_tmp


def aug_2k(dat, meis, rbf_vec, width):
    """
    Build CNN input tensor and loss evaluation tensor for the cXVAE.

    Interleaves log(Z_kt) and MEI values in a 2K-wide spatial layout,
    then stacks [t-1, t, t+1] channels as pseudo-replicates.

    Args:
        dat:     Tensor (K x n_t) — latent expPS variables
        meis:    Tensor (1 x n_t) — ENSO condition values
        rbf_vec: Tensor (1 x K*n_basis) — flattened RBF basis values
        width:   int — knot grid width (sqrt(K))

    Returns:
        dat_torch_enso_3d: Tensor (n_t x 3 x K/width x 2*width) — CNN input
        dat_3d:            Tensor (n_t x 3 x K+n_basis+1)        — loss input
    """
    n_knots, n_t = dat.shape

    # Interleave log(Z) and MEI into 2K channels
    dat_torch_enso = torch.zeros(n_t, 1, int(n_knots / width) * width * 2)
    dat_torch_enso[:, :, 0::2] = dat.T.unsqueeze(1).log()
    dat_torch_enso[:, :, 1::2] = torch.repeat_interleave(meis, n_knots).reshape(-1, n_knots).unsqueeze(1)
    dat_torch_enso = dat_torch_enso.reshape(n_t, 1, -1, width * 2)

    # Stack [t-1, t, t+1] channels
    dat_torch_enso_3d = torch.zeros(n_t, 3, int(n_knots / width), width * 2)
    for t in range(n_t):
        t_prev = t if t == 0 else t - 1
        t_next = t if t == n_t - 1 else t + 1
        dat_torch_enso_3d[t, 0] = dat_torch_enso[t_prev, 0]
        dat_torch_enso_3d[t, 1] = dat_torch_enso[t, 0]
        dat_torch_enso_3d[t, 2] = dat_torch_enso[t_next, 0]

    # Build loss evaluation tensor: [log(Z); RBF_vec; MEI]
    dat_extended = torch.cat([
        dat.log(),
        rbf_vec.repeat(n_t, 1).T,
        meis.T
    ], dim=0)

    dat_3d = torch.zeros(n_t, 3, rbf_vec.shape[1] + n_knots + 1)
    for t in range(n_t):
        t_prev = t if t == 0 else t - 1
        t_next = t if t == n_t - 1 else t + 1
        dat_3d[t, 0] = dat_extended[:, t_prev]
        dat_3d[t, 1] = dat_extended[:, t]
        dat_3d[t, 2] = dat_extended[:, t_next]

    return dat_torch_enso_3d, dat_3d


def z_chess_aug_2k(z, label, width):
    """
    Interleave latent variables z and condition label for decoder input.

    Args:
        z:     Tensor (batch x 3K) — latent variables
        label: Tensor (batch x 3)  — condition values [t-1, t, t+1]
        width: int — knot grid width

    Returns:
        Tensor (batch x 3 x K/width x 2*width) — decoder CNN input
    """
    k = int(z.shape[1] / 3)
    z = z.reshape(-1, 3, k)
    z_4d = torch.zeros([z.shape[0], 3, k * 2])
    for t in range(z.shape[1]):
        z_4d[:, t, 0::2] = z[:, t, :]
        z_4d[:, t, 1::2] = label[:, t].repeat_interleave(k).reshape(z.shape[0], -1)
    z_4d = z_4d.reshape(z.shape[0], 3, -1, width * 2)
    return z_4d


# =============================================================================
# Spatial Basis Functions
# =============================================================================

def wendland(d, r):
    """
    Wendland compactly-supported basis function (s=2, k=1).

    Args:
        d: Tensor of pairwise distances (non-negative)
        r: float — support radius

    Returns:
        Tensor of basis function values (zero beyond radius r)
    """
    out = ((1 - d / r) ** 4) * (4 * d / r + 1)
    out[d >= r] = 0.0
    return out


# =============================================================================
# Dataset Classes
# =============================================================================

class CustomDataset(Dataset):
    """Dataset for CNN pretraining: returns (CNN input, loss evaluation input)."""

    def __init__(self, dat1, dat2):
        self.x = dat1   # CNN input:  (n_t x 3 x H x W)
        self.y = dat2   # Loss input: (n_t x 3 x features)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CVAEinput_Dataset(Dataset):
    """Dataset for cXVAE training: returns (spatial field input, condition input)."""

    def __init__(self, dat1, dat2):
        self.x = dat1   # Spatial field: (n_t x 3*n_loc)
        self.y = dat2   # Condition:     (n_t x 3)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def empirical_cdf_2d(x):
    """
    Compute empirical CDF row-wise for a 2D tensor.

    Args:
        x: Tensor (n_loc x n_time)

    Returns:
        Tensor of same shape with CDF values in [0, 1]
    """
    sorted_x, indices = torch.sort(x, dim=1)
    n_elements = x.shape[1]
    ranks = torch.arange(1, n_elements + 1, dtype=torch.float32, device=x.device)
    ranks = ranks.unsqueeze(0).expand(x.shape[0], -1)
    cdf_values = ranks / n_elements
    inverse_indices = torch.argsort(indices, dim=1)
    return cdf_values.gather(1, inverse_indices)


def chi_est(Data, Loc, d, tol=1e-2, gridded=False,
            u_vec=torch.cat([torch.arange(0.95, 0.98, 0.001),
                             torch.arange(0.981, 0.999, 0.0001)]),
            CDF_fun=None):
    """
    Estimate the chi-coefficient of extremal dependence at spatial lag d.

    Args:
        Data:    Tensor (n_loc x n_time) — spatial process values
        Loc:     Tensor (n_loc x 2)      — spatial coordinates
        d:       float — target spatial lag distance
        tol:     float — distance tolerance for pair selection
        gridded: bool  — whether locations are on a regular grid
        u_vec:   Tensor — quantile levels to evaluate
        CDF_fun: callable (optional) — custom CDF transformation

    Returns:
        dict with keys 'u', 'truth', 'upper', 'lower'
    """
    import random

    Dist = torch.cdist(Loc, Loc, p=2)
    mask = (Dist > (d - tol)) & (Dist < (d + tol))
    pairs = torch.nonzero(mask)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    if (len(pairs) * Data.shape[1]) > 500:
        dx = Loc[pairs[:, 1], 0] - Loc[pairs[:, 0], 0]
        dy = Loc[pairs[:, 1], 1] - Loc[pairs[:, 0], 1]
        pairs = pairs[(dx > 0) & (dy > 0)]

    if gridded and (len(pairs) > 500):
        pairs = pairs[::2]

    if len(pairs) > 100:
        idx = random.sample(range(len(pairs)), 100)
        pairs = pairs[idx]

    n_times = Data.shape[1]
    expanded_pairs = pairs.repeat(n_times, 1)
    times = torch.arange(n_times).repeat_interleave(len(pairs))

    dep_pairs = torch.stack([
        Data[expanded_pairs[:, 0], times],
        Data[expanded_pairs[:, 1], times]
    ], dim=1)

    if CDF_fun:
        U_pairs = CDF_fun(dep_pairs)
    else:
        U_pairs = torch.zeros_like(dep_pairs)
        for col in [0, 1]:
            sorted_vals, _ = torch.sort(dep_pairs[:, col])
            ranks = torch.searchsorted(sorted_vals, dep_pairs[:, col], right=True)
            U_pairs[:, col] = ranks.float() / len(dep_pairs[:, col])

    Min_sim = torch.min(U_pairs, dim=1)[0]
    results = torch.zeros((len(u_vec), 3))

    for i, u in enumerate(u_vec):
        num_tmp = (Min_sim > u).float()
        denom_tmp = (U_pairs[:, 0] > u).float()
        p_sim = num_tmp.mean()
        p1_sim = denom_tmp.mean()

        if p1_sim == 0 or p_sim == 0:
            results[i] = torch.tensor([0.0, 0.0, 0.0])
        else:
            ratio = p_sim / p1_sim
            var_A = p_sim * (1 - p_sim) / len(num_tmp)
            var_B = p1_sim * (1 - p1_sim) / len(denom_tmp)
            cov_AB = ((num_tmp - p_sim) * (denom_tmp - p1_sim)).mean() / len(num_tmp)
            var_r = (var_A / p1_sim**2
                     + p_sim**2 * var_B / p1_sim**4
                     - 2 * p_sim * cov_AB / p1_sim**3)
            std_sim = torch.sqrt(var_r.clamp(min=0))
            results[i] = torch.tensor([
                (ratio - 1.96 * std_sim).clamp(0, 1),
                (ratio + 1.96 * std_sim).clamp(0, 1),
                ratio
            ])

    return {
        'u':     u_vec.cpu().numpy(),
        'truth': results[:, 2].cpu().numpy(),
        'upper': results[:, 1].cpu().numpy(),
        'lower': results[:, 0].cpu().numpy()
    }


def ARE(u, unit_area, x, s0_idx=272):
    """
    Compute the Averaged Radius of Exceedances (ARE) at quantile u.

    See Zhang et al. (2022) for definition and asymptotic properties.

    Args:
        u:         float — quantile threshold in (0, 1)
        unit_area: float — area of each grid cell (psi^2)
        x:         Tensor (n_loc x n_time) — spatial process
        s0_idx:    int — index of reference location s_0

    Returns:
        Tensor [ARE_estimate, lower_95CI, upper_95CI]
    """
    s_0 = torch.tensor([s0_idx])
    U_ir = empirical_cdf_2d(x)

    _, where_exceed = torch.where(U_ir[s_0, :] > u)

    if where_exceed.shape[0] == 0:
        return torch.tensor([0.0, 0.0, 0.0])

    tmp_AE = torch.zeros(where_exceed.shape[0])
    for col in range(where_exceed.shape[0]):
        tmp_AE[col] = torch.sum(U_ir[:, where_exceed[col]] > u).mul(unit_area)

    truth = torch.median(tmp_AE).div(torch.pi).sqrt()
    q025, q975 = torch.quantile(tmp_AE, torch.tensor([0.025, 0.975]))
    lower = truth + q025.div(torch.pi).sqrt().sub(truth).div(2)
    upper = truth + q975.div(torch.pi).sqrt().sub(truth).div(2)

    return torch.tensor([truth, lower, upper])

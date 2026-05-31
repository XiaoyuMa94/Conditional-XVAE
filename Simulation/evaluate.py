# =============================================================================
# evaluate.py
#
# Evaluation and visualization for the simulation study in Ma et al. (2025).
# Computes chi-coefficients, ARE, tail-weighted CRPS, Q-Q plots, and
# spatial field comparisons for both cXVAE and the Sohn cVAE baseline.
#
# Prerequisites:
#   - train_cXVAE.py has been run (cXVAE_trained.pt exists)
#   - baseline_SohnCVAE.py has been run (SohnCVAE_trained.pt exists)
#   - Emulation tensors are available (see "Load Emulations" section)
# =============================================================================

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import utils

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR   = "data/simulation"
FIG_DIR    = "figures/simulation"
os.makedirs(FIG_DIR, exist_ok=True)

ALPHA    = 0.5
MEI_SCALE = 2.0
N_LOC    = 2500
N_T      = 528
ALPHA0   = 30.0   # log-Laplace noise parameter
N_REPS   = 10     # emulation replicates for chi/ARE confidence intervals

# =============================================================================
# Load Data
# =============================================================================

def load_csv(filename):
    return pd.read_csv(
        os.path.join(DATA_DIR, filename),
        header=None, skiprows=1
    ).drop([0], axis="columns").values


X       = load_csv("X_Data.csv")
Thetas  = load_csv("Thetas_Data.csv")
MEIs_MA = load_csv("MEIs_MA_Data.csv") * MEI_SCALE

X_tensor      = torch.tensor(X).to(torch.float32)
Thetas_tensor = torch.tensor(Thetas)
MEI_tensor    = torch.tensor(MEIs_MA).to(torch.float32)

# Spatial locations (50x50 grid over [0,20]x[0,20])
station_x = np.linspace(0, 20, 50)
station_y = np.linspace(0, 20, 50)
xx, yy    = np.meshgrid(station_x, station_y)
stations  = np.column_stack([xx.ravel(), yy.ravel()])
location_tensor = torch.tensor(stations, dtype=torch.float32)

# Holdout locations (extreme-rich, matching train_cXVAE.py selection)
q          = 0.9
threshold  = torch.quantile(X_tensor, q)
count_high = (X_tensor > threshold).sum(dim=1)
_, sorted_idx = torch.sort(count_high)
holdout_idx = sorted_idx[N_LOC - 10:]
X_holdout   = X_tensor[holdout_idx, :]

# =============================================================================
# Load Emulations
# =============================================================================
# Run train_cXVAE.py and baseline_SohnCVAE.py first to produce these tensors.
# Then load or generate them here before running the evaluation code.
#
# Expected shapes:
#   cXVAE_emulated:              (N_LOC x N_T x N_REPS)
#   cXVAE_emulated_counterfact:  (N_LOC x N_T x N_REPS)
#   cXVAE_emulated_whitenoise:   (N_LOC x N_T x N_REPS)
#   Sohn_emulated:               (N_LOC x N_T x N_REPS)
#   Sohn_emulated_counterfact:   (N_LOC x N_T x N_REPS)
#   Sohn_emulated_whitenoise:    (N_LOC x N_T x N_REPS)
# =============================================================================

# =============================================================================
# Section 1: Theta Estimation Maps (Figure 3, rows 2-3)
# =============================================================================

def plot_theta_comparison(theta_pred, Thetas_tensor, target_times,
                          knot_width=15, save_path=None):
    """
    Plot estimated vs true tilting parameter fields at selected time points.

    Args:
        theta_pred:    Tensor (n_t x k) — estimated theta from model
        Thetas_tensor: Tensor (k x n_t) — true theta
        target_times:  list of int — time indices to plot (0-indexed)
        knot_width:    int — knot grid side length
        save_path:     str or None
    """
    n_cols = len(target_times)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    for col, t in enumerate(target_times):
        for row, (data, title) in enumerate([
            (theta_pred[t].reshape(knot_width, knot_width).T.detach().numpy(), "Estimated"),
            (Thetas_tensor[:, t].reshape(knot_width, knot_width).T.detach().numpy(), "True")
        ]):
            im = axes[row, col].imshow(np.flip(data, 0), cmap="plasma", vmin=0, vmax=0.05)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t = {t}", fontsize=14)
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=12)

    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\theta$")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 2: Spatial Field Comparison (Figure 3, rows 4-6)
# =============================================================================

def plot_field_comparison(X_true, X_emu, X_cf,
                          target_times, save_path=None):
    """
    Compare true log(X), emulated log(X), and relative difference
    (counterfactual vs factual) at selected time points.
    """
    n_cols    = len(target_times)
    row_labels = [r"$\log(X)$ (True)", r"$\log(X)$ (Emulated)", "Rel. Diff"]
    vmin, vmax = -4, 2.5

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))

    for col, t in enumerate(target_times):
        true_field = X_true[:, t].log().reshape(50, 50).T.numpy()
        emu_field  = X_emu[:, t].log().reshape(50, 50).T.numpy()
        diff_field = ((X_cf[:, t] - X_emu[:, t]) / X_emu[:, t]).reshape(50, 50).T.numpy()

        for row, (data, cmap, vl, vh) in enumerate([
            (true_field, "PuOr",  vmin, vmax),
            (emu_field,  "PuOr",  vmin, vmax),
            (diff_field, "RdBu_r", -0.5, 0.5)
        ]):
            im = axes[row, col].imshow(np.flip(data, 0), cmap=cmap, vmin=vl, vmax=vh)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t = {t}", fontsize=14)
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=11, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 3: Chi-Coefficient Comparison
# =============================================================================

def compute_chi_comparison(X_true, Emu_data, location_tensor,
                           distance_vec, u_vec, n_reps=N_REPS):
    """
    Estimate chi-coefficient for truth and emulation at multiple spatial lags.

    Returns:
        dict keyed by distance, each containing truth and emulation curves
    """
    comparisons = {}

    for d in distance_vec:
        d = float(d)
        chi_truth = utils.chi_est(
            Data=X_true, Loc=location_tensor,
            d=d, tol=0.1, gridded=True, u_vec=u_vec
        )
        chi_reps = torch.zeros(n_reps, 3, len(u_vec))

        for rep in range(n_reps):
            print(f"  Chi rep {rep + 1}/{n_reps}, distance={d}")
            tmp = utils.chi_est(
                Data=Emu_data[:, :, rep], Loc=location_tensor,
                d=d, tol=0.1, gridded=True, u_vec=u_vec
            )
            chi_reps[rep, 0] = torch.tensor(tmp["truth"])
            chi_reps[rep, 1] = torch.tensor(tmp["upper"])
            chi_reps[rep, 2] = torch.tensor(tmp["lower"])

        comparisons[d] = {**chi_truth,
                          "emu":       chi_reps[:, 0].mean(0),
                          "emu_upper": chi_reps[:, 1].mean(0),
                          "emu_lower": chi_reps[:, 2].mean(0)}
    return comparisons


def plot_chi(comparisons, distance_labels, title_label, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    linestyles = ["-", "--", ":"]

    for i, (d, res) in enumerate(sorted(comparisons.items())):
        ls = linestyles[i % 3]
        plt.plot(res["u"], res["truth"], color="r", linestyle=ls, linewidth=1.5)
        plt.fill_between(res["u"], res["lower"], res["upper"], color="red", alpha=0.1)
        plt.plot(res["u"], res["emu"].numpy(), color="b", linestyle=ls, linewidth=1.5)
        plt.fill_between(res["u"], res["emu_lower"].numpy(),
                         res["emu_upper"].numpy(), color="blue", alpha=0.1)

    for label, ypos in zip(distance_labels, [0.78, 0.35, 0.10]):
        plt.text(0.15, ypos, label, fontsize=12, transform=plt.gca().transAxes)

    plt.ylim(0, 1)
    plt.xlabel("Quantile (u)", fontsize=16)
    plt.ylabel(r"$\chi(u)$", fontsize=16)
    plt.tick_params(labelsize=14)
    fig.text(0.01, 0.5, title_label, fontsize=14, fontweight="bold",
             va="center", rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 4: Averaged Radius of Exceedances (ARE)
# =============================================================================

def compute_ARE(X_true, Emu_data, u_vec, unit_area, n_reps=N_REPS):
    """Compute ARE for truth and emulation across quantile levels."""
    ARE_truth = torch.zeros(3, len(u_vec))
    for i, u in enumerate(u_vec):
        ARE_truth[:, i] = utils.ARE(u, unit_area=unit_area, x=X_true)

    ARE_reps = torch.zeros(n_reps, 3, len(u_vec))
    for rep in range(n_reps):
        print(f"  ARE rep {rep + 1}/{n_reps}")
        for i, u in enumerate(u_vec):
            ARE_reps[rep, :, i] = utils.ARE(u, unit_area=unit_area,
                                             x=Emu_data[:, :, rep])

    ARE_mean = ARE_reps[:, 0].mean(0)
    ARE_low  = ARE_reps[:, 1].mean(0)
    ARE_high = ARE_reps[:, 2].mean(0)

    return ARE_truth, ARE_mean, ARE_low, ARE_high


def plot_ARE(u_vec, ARE_truth, ARE_mean, ARE_low, ARE_high, save_path=None):
    plt.figure(figsize=(8, 5))
    u = u_vec[:-5].numpy()
    plt.plot(u, ARE_truth[0, :-5], "r-", label="Truth")
    plt.fill_between(u, ARE_truth[1, :-5], ARE_truth[2, :-5],
                     color="tomato", alpha=0.5, label="95% CI (Truth)")
    plt.plot(u, ARE_mean[:-5], "b-", label="Emulation")
    plt.fill_between(u, ARE_low[:-5], ARE_high[:-5],
                     color="lightblue", alpha=0.5, label="95% CI (Emulation)")
    plt.xlabel("Quantile", fontsize=18)
    plt.ylabel("Averaged Radius of Exceedances", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 5: Tail-Weighted CRPS
# =============================================================================

def compute_tail_weighted_crps(X_target, emulation, tail_quantile=0.9):
    """
    Compute tail-weighted CRPS at each (location, time) using upper-tail weight.

    Args:
        X_target:  Tensor (n_holdout x n_time)
        emulation: Tensor (n_holdout x n_time x n_sample)
        tail_quantile: float — weight = 1 if z > q-th percentile, else 0

    Returns:
        CRPS: Tensor (n_holdout x n_time)
    """
    n_loc, n_time, n_sample = emulation.shape
    CRPS = torch.zeros(n_loc, n_time)

    for loc in range(n_loc):
        for t in range(n_time):
            samples  = torch.sort(emulation[loc, t])[0]
            z_tail   = samples[int(n_sample * tail_quantile)]
            upper    = torch.max(torch.cat([samples, X_target[loc, t].unsqueeze(0)])) + 1e-4
            z_grid   = torch.linspace(0.0, upper.item(), 1000)
            ecdf     = torch.tensor([torch.mean((samples <= z).float()) for z in z_grid])
            indicator = (X_target[loc, t] <= z_grid).float()
            weight    = (z_grid > z_tail).float()
            dz        = z_grid[1] - z_grid[0]
            CRPS[loc, t] = torch.trapz(weight * (ecdf - indicator) ** 2, dx=dz)

        if (loc + 1) % 2 == 0:
            print(f"  CRPS: location {loc + 1}/{n_loc}")

    return CRPS


def plot_crps_violin(crps_list, labels, save_path=None):
    """Violin plot of tail-weighted CRPS (log scale) across models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data  = [c.log().view(-1).numpy() for c in crps_list]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("skyblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)
    parts["cmedians"].set_color("darkblue")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("Tail-weighted CRPS (log scale)", fontsize=18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 6: Q-Q Plots
# =============================================================================

def qq_plot(x, y, lim=0.4, fontsize=18, xlabel="Truth",
            ylabel="Emulation", save_path=None):
    """
    Q-Q plot with 95% Kolmogorov-Smirnov confidence bands.

    Args:
        x: Tensor — true observations
        y: Tensor — emulated values
        lim: float — axis limit
    """
    mask    = ~torch.isnan(x) & ~torch.isnan(y)
    x, y    = x[mask], y[mask]
    n       = 200
    p       = torch.linspace(1e-4, 1 - 1e-4, n)
    xq      = torch.quantile(x, p)
    yq      = torch.quantile(y, p)
    K       = 1.36
    M       = len(x) * len(y) / (len(x) + len(y))
    yl      = torch.quantile(y, torch.clamp(p - K / np.sqrt(M), 0, 1))
    yu      = torch.quantile(y, torch.clamp(p + K / np.sqrt(M), 0, 1))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xq, yq, s=20, color="black")
    ax.plot(xq, yl, "--", color="gray",       label="95% band")
    ax.plot(xq, yu, "--", color="gray")
    ax.plot(xq, xq, "--", color="darkorange", label="1-1 line")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.legend(fontsize=fontsize - 2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(True, linestyle=":")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Section 7: Counterfactual Kernel Density Plots
# =============================================================================

def plot_kernel_density(factual_s1, factual_s2, counterfact_s1, counterfact_s2,
                        label_factual="cXVAE Emulation", label_cf="Counterfactual",
                        xlim=None, ylim=None, save_path=None):
    """
    2D kernel density contour plot comparing factual and counterfactual emulations
    at two spatial locations.
    """
    def kde_grid(s1, s2, n=100):
        x = np.linspace(float(min(s1)), float(max(s1)), n)
        y = np.linspace(float(min(s2)), float(max(s2)), n)
        X, Y = np.meshgrid(x, y)
        Z = np.reshape(
            gaussian_kde(np.vstack([s1, s2]), bw_method=0.8)(
                np.vstack([X.ravel(), Y.ravel()])
            ).T, X.shape
        )
        return X, Y, Z / Z.max()

    X1, Y1, Z1 = kde_grid(factual_s1,    factual_s2)
    X2, Y2, Z2 = kde_grid(counterfact_s1, counterfact_s2)
    levels      = np.linspace(0.1, 1, 10)

    fig, ax = plt.subplots(figsize=(8, 8))
    c1 = ax.contour(X1, Y1, Z1, levels=levels, colors="red",  linewidths=1)
    c2 = ax.contour(X2, Y2, Z2, levels=levels, colors="blue", linewidths=1)
    plt.clabel(c1, inline=True, fontsize=10, fmt="%.1f")
    plt.clabel(c2, inline=True, fontsize=10, fmt="%.1f")

    legend_elements = [
        Line2D([0], [0], color="red",  lw=1, label=label_factual),
        Line2D([0], [0], color="blue", lw=1, label=label_cf)
    ]
    ax.legend(handles=legend_elements, fontsize=18)
    ax.set_xlabel(r"$X(\mathbf{s}_1)$", fontsize=22)
    ax.set_ylabel(r"$X(\mathbf{s}_2)$", fontsize=22)
    ax.tick_params(labelsize=17)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# Run All Evaluations
# =============================================================================
# Uncomment blocks below after loading your emulation tensors.
# =============================================================================

# --- Chi-coefficient ---
# u_vec        = torch.cat([torch.arange(0.95, 0.98, 0.001),
#                            torch.arange(0.981, 0.999, 0.0001)])
# distance_vec = [0.5, 3.0, 6.0]
# unit_area    = 400 / 2401
#
# chi_cXVAE = compute_chi_comparison(
#     X_tensor, cXVAE_emulated, location_tensor,
#     distance_vec, u_vec, n_reps=N_REPS
# )
# plot_chi(chi_cXVAE,
#          distance_labels=["Short-range", "Medium-range", "Long-range"],
#          title_label="cXVAE Model",
#          save_path=os.path.join(FIG_DIR, "chi_cXVAE.png"))
#
# chi_Sohn = compute_chi_comparison(
#     X_tensor, Sohn_emulated, location_tensor,
#     distance_vec, u_vec, n_reps=N_REPS
# )
# plot_chi(chi_Sohn,
#          distance_labels=["Short-range", "Medium-range", "Long-range"],
#          title_label="Vanilla cVAE Model",
#          save_path=os.path.join(FIG_DIR, "chi_SohnCVAE.png"))

# --- ARE ---
# u_vec_are = torch.cat([torch.arange(0, 0.99, 0.01),
#                         torch.arange(0.981, 0.999, 0.001)])
#
# ARE_truth, ARE_mean_cx, ARE_low_cx, ARE_high_cx = compute_ARE(
#     X_tensor, cXVAE_emulated, u_vec_are, unit_area
# )
# plot_ARE(u_vec_are, ARE_truth, ARE_mean_cx, ARE_low_cx, ARE_high_cx,
#          save_path=os.path.join(FIG_DIR, "ARE_cXVAE.png"))
#
# _, ARE_mean_sohn, ARE_low_sohn, ARE_high_sohn = compute_ARE(
#     X_tensor, Sohn_emulated, u_vec_are, unit_area
# )
# plot_ARE(u_vec_are, ARE_truth, ARE_mean_sohn, ARE_low_sohn, ARE_high_sohn,
#          save_path=os.path.join(FIG_DIR, "ARE_SohnCVAE.png"))

# --- CRPS ---
# Emu_holdout_cXVAE    = cXVAE_emulated[holdout_idx, :, :]
# Emu_holdout_white_cx = cXVAE_emulated_whitenoise[holdout_idx, :, :]
# Emu_holdout_Sohn     = Sohn_emulated[holdout_idx, :, :]
# Emu_holdout_white_s  = Sohn_emulated_whitenoise[holdout_idx, :, :]
#
# CRPS_cx    = compute_tail_weighted_crps(X_holdout, Emu_holdout_cXVAE)
# CRPS_cx_wn = compute_tail_weighted_crps(X_holdout, Emu_holdout_white_cx)
# CRPS_sohn  = compute_tail_weighted_crps(X_holdout, Emu_holdout_Sohn)
# CRPS_sohn_wn = compute_tail_weighted_crps(X_holdout, Emu_holdout_white_s)
#
# plot_crps_violin(
#     [CRPS_cx, CRPS_sohn, CRPS_cx_wn, CRPS_sohn_wn],
#     labels=["cXVAE", "Vanilla cVAE", "cXVAE*", "Vanilla cVAE*"],
#     save_path=os.path.join(FIG_DIR, "crps_violin.pdf")
# )

# --- Q-Q plot ---
# high_enso_t = int(torch.argmax(MEI_tensor).item())
# qq_plot(X_tensor[:, high_enso_t],
#         cXVAE_emulated[:, high_enso_t, 0],
#         lim=0.4, ylabel="cXVAE Emulation",
#         save_path=os.path.join(FIG_DIR, "qq_cXVAE.pdf"))
# qq_plot(X_tensor[:, high_enso_t],
#         Sohn_emulated[:, high_enso_t, 0],
#         lim=0.4, ylabel="Vanilla cVAE Emulation",
#         save_path=os.path.join(FIG_DIR, "qq_SohnCVAE.pdf"))

# --- Counterfactual kernel density ---
# t_idx = 228  # December 1997 (high El Nino)
# plot_kernel_density(
#     factual_s1    = cXVAE_emulated.reshape(50, 50, N_T, -1)[22, 8, t_idx],
#     factual_s2    = cXVAE_emulated.reshape(50, 50, N_T, -1)[18, 14, t_idx],
#     counterfact_s1 = cXVAE_emulated_counterfact.reshape(50, 50, N_T, -1)[22, 8, t_idx],
#     counterfact_s2 = cXVAE_emulated_counterfact.reshape(50, 50, N_T, -1)[18, 14, t_idx],
#     save_path=os.path.join(FIG_DIR, f"kernel_cXVAE_t{t_idx}.png")
# )

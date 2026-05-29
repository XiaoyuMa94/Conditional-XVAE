# =============================================================================
# train_cXVAE_FWI.py
#
# Train the conditional XVAE (cXVAE) on the Fire Weather Index (FWI)
# real data analysis in Ma et al. (2025).
#
# Prerequisites:
#   - data_preparation.R has been run and CSV outputs exist in DATA_DIR
#   - pretrain_CNN_FWI.py has been run and CNN_pretrained_FWI.pt exists
#
# Output:
#   cXVAE_FWI_trained.pt  — saved model weights (best validation epoch)
# =============================================================================

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import utils

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR       = "data/FWI"
PRETRAINED_CNN = "CNN_pretrained_FWI.pt"
MODEL_OUT      = "cXVAE_FWI_trained.pt"
FIG_DIR        = "figures/FWI"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FIG_DIR, exist_ok=True)

# Hyperparameters
ALPHA         = 0.5
ALPHA0        = 30.0    # log-Laplace noise parameter
MEI_SCALE     = 2.0
LEARNING_RATE = 1e-7
NUM_EPOCHS    = 600
BATCH_SIZE    = 127
VAL_RATIO     = 0.1
PATIENCE      = 20
MIN_DELTA     = 1e-4

# Spatial domain (eastern Australia)
LON_MIN, LON_MAX = 143.125, 150.9375
LAT_MIN, LAT_MAX = -33.75,  -23.25

# =============================================================================
# Load Data
# =============================================================================

def load_csv(filename):
    return pd.read_csv(
        os.path.join(DATA_DIR, filename),
        header=None, skiprows=1
    ).drop([0], axis="columns").values


X   = load_csv("X_Data.csv")
W   = load_csv("W_Data.csv")
RBF = load_csv("RBF_Data.csv")

Nina34 = torch.tensor(
    pd.read_csv(os.path.join(DATA_DIR, "MEIs_Data.csv")).to_numpy()
)[:, 1]
Nina34 = Nina34.reshape(127, 1) * MEI_SCALE   # (127, 1)

GEV_par = torch.tensor(load_csv("GEV_par.csv"))  # (n_loc x 3): location, scale, shape

X_tensor  = torch.from_numpy(X).to(torch.float32)
W_alpha   = torch.tensor(W).pow(1 / ALPHA).to(torch.float32)
rbf_mat   = torch.tensor(RBF).to(torch.float32)

k          = W_alpha.shape[1]       # 540
knot_width = 20                     # 20x27 knot grid
alpha_size = rbf_mat.shape[1]
n_loc      = X_tensor.shape[0]      # 1118
n_t        = X_tensor.shape[1]      # 127
latent_size = k * 3
hidden_size = k * 3
image_size  = n_loc * 3             # flattened [t-1, t, t+1] spatial fields

# =============================================================================
# Holdout Location Selection (extreme-rich locations)
# =============================================================================

q           = 0.9
threshold   = torch.quantile(X_tensor, q)
count_high  = (X_tensor > threshold).sum(dim=1)
_, sorted_idx = torch.sort(count_high)

n_holdout   = 10
holdout_idx = sorted_idx[n_loc - n_holdout:]
train_idx   = np.setdiff1d(np.arange(n_loc), holdout_idx.cpu().numpy())

station_x = np.linspace(LON_MIN, LON_MAX, 26)
station_y = np.linspace(LAT_MIN, LAT_MAX, 43)
xx, yy    = np.meshgrid(station_x, station_y)
stations  = np.column_stack([xx.ravel(), yy.ravel()])

X_train   = X_tensor
X_holdout = X_tensor[holdout_idx, :]
W_train   = W_alpha

# =============================================================================
# Encoder Initialization via Truncated SVD
# =============================================================================

U, D, Vt = np.linalg.svd(W_alpha.numpy(), full_matrices=False)
keep      = D > (1e-2 * np.max(D))
U_k, D_k, V_k = U[:, keep], D[keep], Vt[keep, :].T
proj      = torch.tensor(V_k @ np.diag(1 / D_k) @ U_k.T).T.to(torch.float32)

# Block-diagonal projection and W_alpha matrices for encoder/decoder init
def build_block_diag(M):
    z = torch.zeros_like(M)
    return torch.cat([
        torch.cat([M, z, z]),
        torch.cat([z, M, z]),
        torch.cat([z, z, M])
    ], dim=1)

proj_final    = build_block_diag(proj.T)
W_alpha_final = build_block_diag(W_train.T)

# =============================================================================
# Dataset and DataLoader
# =============================================================================

X_train_tensor = utils.x_aug(X_train)
Nina34_input   = utils.x_aug(Nina34.T)

full_dataset  = utils.CVAEinput_Dataset(X_train_tensor, Nina34_input)
n_val         = int(VAL_RATIO * len(full_dataset))
n_train       = len(full_dataset) - n_val

train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# CNN Submodule (same architecture as pretrain_CNN_FWI.py)
# =============================================================================

class CNN(nn.Module):
    def __init__(self, alpha_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(128, 64,  kernel_size=3, padding="same")
        self.conv7 = nn.Conv2d(64, 32,   kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(32, 16,   kernel_size=3, padding="same")
        self.pool  = nn.MaxPool2d(2, 1)

        rbf_vec   = rbf_mat.reshape(1, -1)
        Z_dummy   = torch.ones(k, 1)
        mei_dummy = torch.ones(1, 1)
        dummy_3d, _ = utils.aug_2k(Z_dummy, mei_dummy, rbf_vec, knot_width)
        with torch.no_grad():
            flat_dim = self._forward_conv(dummy_3d[:1]).view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, alpha_size)

    def _forward_conv(self, x):
        x = self.pool(self.conv2(self.pool(self.conv1(x))))
        x = self.pool(self.conv4(self.pool(self.conv3(x))))
        x = self.pool(self.conv6(self.pool(self.conv5(x))))
        x = self.pool(self.conv8(self.pool(self.conv7(x))))
        return x

    def forward(self, x):
        return self.fc1(torch.flatten(self._forward_conv(x), start_dim=1)).view(-1, alpha_size)


# =============================================================================
# cXVAE Model
# =============================================================================

class CVAE(nn.Module):
    """
    Conditional XVAE for FWI real data analysis.
    Same architecture as simulation but adapted for:
      - n_loc = 1118 locations
      - k = 540 knots (20x27 grid)
      - n_t = 127 monthly time points
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.fc1      = nn.Linear(image_size, hidden_size)
        self.fc2      = nn.Linear(hidden_size, latent_size)
        self.fc3      = nn.Linear(hidden_size, latent_size)
        self.softplus = nn.Softplus(beta=8)

        # CNN decoder (for theta estimation)
        self.conv1 = nn.Conv2d(3, 32,  kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(128, 64,  kernel_size=3, padding="same")
        self.conv7 = nn.Conv2d(64, 32,   kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(32, 16,   kernel_size=3, padding="same")
        self.pool  = nn.MaxPool2d(2, 1)

        # Infer CNN output size
        rbf_vec   = rbf_mat.reshape(1, -1)
        Z_dummy   = torch.ones(k, 1)
        mei_dummy = torch.ones(1, 1)
        dummy_3d, _ = utils.aug_2k(Z_dummy, mei_dummy, rbf_vec, knot_width)
        with torch.no_grad():
            flat_dim = self._forward_cnn(dummy_3d[:1]).view(1, -1).shape[1]
        self.fct = nn.Linear(flat_dim, alpha_size)

        # ENSO linear injection: g(c_t) = b * c_t
        self.reg_lin = nn.Linear(3, latent_size, bias=False)

        # Spatial decoder W: learnable basis mapping Z_t -> Y_t
        self.l1 = nn.Linear(latent_size * 2, image_size)

    def _forward_cnn(self, x):
        x = self.pool(self.conv2(self.pool(self.conv1(x))))
        x = self.pool(self.conv4(self.pool(self.conv3(x))))
        x = self.pool(self.conv6(self.pool(self.conv5(x))))
        x = self.pool(self.conv8(self.pool(self.conv7(x))))
        return x

    def encode(self, x):
        h1  = self.softplus(self.fc1(x)) + 1e-5
        mu  = self.softplus(self.fc2(h1))
        var = torch.exp(self.fc3(h1))
        return mu, var

    def reparameterize(self, mu, var, label):
        eps      = torch.randn(var.shape, device=var.device)
        reg_term = self.reg_lin(label)
        z        = mu.log() + torch.sqrt(var) * eps + reg_term
        return z, eps

    def decode(self, z, label):
        z_inputs = utils.z_chess_aug_2k(z, label, width=knot_width).to(self.conv1.weight.device)

        # CNN: estimate basis coefficients for theta
        t1 = self._forward_cnn(z_inputs)
        random_coefficients = self.fct(torch.flatten(t1, start_dim=1)).view(-1, alpha_size)

        # Spatial decoder: W * z -> Y_t
        z_origin = z_inputs.exp().reshape(z_inputs.shape[0], -1)
        y_star   = self.softplus(self.l1(z_origin))

        return y_star, random_coefficients

    def forward(self, x, label):
        mu, var                     = self.encode(x)
        z, eps                      = self.reparameterize(mu, var, label)
        y_star, random_coefficients = self.decode(z, label)
        return y_star, random_coefficients, eps, var, z


# =============================================================================
# Model Initialization
# =============================================================================

torch.manual_seed(12)
model = CVAE().to(DEVICE)

# Initialize spatial decoder W with Wendland basis
mask = torch.zeros_like(model.l1.weight)
mask[:, 0::2] = W_alpha_final.T
model.l1.weight  = nn.Parameter(mask)
model.l1.bias    = nn.Parameter(torch.zeros_like(model.l1.bias))

# Initialize encoder with SVD-based projection
model.fc1.weight = nn.Parameter(proj_final)
model.fc1.bias   = nn.Parameter(torch.zeros_like(model.fc1.bias))
model.fc2.weight = nn.Parameter(torch.diag(torch.ones(model.fc2.weight.shape[0])))
model.fc2.bias   = nn.Parameter(torch.full_like(model.fc2.bias, 1e-4))
model.fc3.weight = nn.Parameter(torch.zeros_like(model.fc3.weight))
model.fc3.bias   = nn.Parameter(torch.full_like(model.fc3.bias, -10.0))
model.reg_lin.weight = nn.Parameter(torch.full_like(model.reg_lin.weight, 0.01))

# Load pretrained CNN weights (recommended)
if os.path.exists(PRETRAINED_CNN):
    pretrained = CNN(alpha_size=alpha_size).to(DEVICE)
    pretrained.load_state_dict(torch.load(PRETRAINED_CNN, map_location=DEVICE))
    for layer in ["conv1", "conv2", "conv3", "conv4", "conv5",
                  "conv6", "conv7", "conv8"]:
        getattr(model, layer).load_state_dict(getattr(pretrained, layer).state_dict())
    model.fct.load_state_dict(pretrained.fc1.state_dict())
    print(f"Loaded pretrained CNN weights from {PRETRAINED_CNN}")
else:
    print(f"Warning: {PRETRAINED_CNN} not found. Training from scratch (not recommended).")

# =============================================================================
# Loss Function (ELBO)
# =============================================================================

def loss_function(x, label, var, eps, y_star, rbf_mat, random_coefficients, z):
    """
    Negative ELBO for the FWI cXVAE:
      loss_p1: log-Laplace data likelihood
      loss_p2: expPS prior on latent variables
      loss_p3: KL divergence
      loss_p4: temporal smoothness penalty on xi_t
    """
    # Part 1: log-Laplace likelihood
    standardized = x.div(y_star + 1e-3)
    loss_p1 = (standardized.log().abs().mul(-ALPHA0) - standardized.log())
    loss_p1 = loss_p1.sum(dim=1).mean()

    # Part 2: expPS log-likelihood
    theta        = torch.mm(rbf_mat, random_coefficients.T).relu()
    z_reshaped   = z.reshape(-1, k).T
    log_lik_z    = (theta.sqrt()
                    - z_reshaped.mul(1.5)
                    - z_reshaped.exp().mul(theta)
                    - z_reshaped.exp().mul(4).pow(-1))
    loss_p2 = log_lik_z.sum().div(log_lik_z.numel())

    # Part 3: KL divergence
    loss_p3 = (var.log().sum(dim=1) + eps.pow(2).sum(dim=1) * 0.5).mean().div(3)

    # Part 4: temporal smoothness on basis coefficients
    r_tmp   = random_coefficients.reshape(-1, 3, alpha_size)
    d1      = torch.diff(r_tmp, dim=1)
    ts      = torch.diff(label, dim=1).abs().clamp(min=3e-4)
    loss_p4 = -0.001 * d1.div(ts.repeat_interleave(alpha_size).reshape(-1, 2, alpha_size)
                               ).pow(2).sum().sqrt().div(d1.numel())

    return loss_p1, loss_p2, loss_p3, loss_p4


# =============================================================================
# Optimizer (layer-specific learning rates)
# =============================================================================

optimizer = optim.Adam([
    {"params": [p for n, p in model.named_parameters() if n != "l1.weight"],
     "lr": LEARNING_RATE},
    {"params": [model.l1.weight],
     "lr": 5e-6},   # spatial decoder W — main learnable component
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-10
)

# =============================================================================
# Training Loop with Early Stopping
# =============================================================================

best_val_loss    = float("inf")
best_epoch       = 0
epochs_no_improve = 0
best_model_state  = None
loss_list = []

start_time = time.time()
print(f"Training cXVAE (FWI) on {DEVICE}...")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    t = tqdm(enumerate(train_loader), total=len(train_loader),
             desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for i, (data, label) in t:
        data, label = data.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()

        y_star, random_coefficients, eps, var, z = model(data, label)
        loss_p1, loss_p2, loss_p3, loss_p4 = loss_function(
            data, label, var, eps, y_star, rbf_mat, random_coefficients, z
        )
        loss = (loss_p1 + loss_p2 + loss_p3 + loss_p4).mul(-1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        t.set_description(
            f"Epoch {epoch + 1} | Loss: {loss.item():.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            y_star, random_coefficients, eps, var, z = model(data, label)
            loss_p1, loss_p2, loss_p3, loss_p4 = loss_function(
                data, label, var, eps, y_star, rbf_mat, random_coefficients, z
            )
            val_loss += (loss_p1 + loss_p2 + loss_p3 + loss_p4).mul(-1).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    loss_list.append(train_loss)

    print(f"Epoch {epoch + 1}: Train = {train_loss:.4f} | Val = {val_loss:.4f}")

    if val_loss < best_val_loss * (1 - MIN_DELTA):
        best_val_loss    = val_loss
        best_epoch       = epoch + 1
        epochs_no_improve = 0
        best_model_state = {k_: v.cpu() for k_, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}.")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best model from epoch {best_epoch}.")

elapsed = time.time() - start_time
print(f"Training complete in {elapsed:.2f} seconds.")
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# =============================================================================
# Emulation and Back-transformation
# =============================================================================

def emulate(model, condition_data, n_samples=500):
    """
    Generate emulated spatial fields from the learned prior p(z | c).

    Returns:
        Tensor (n_samples x n_t x n_loc)
    """
    model.eval()
    device     = next(model.parameters()).device
    cond_input = condition_data.to(device)

    full_data = utils.CVAEinput_Dataset(X_train_tensor, cond_input)
    Emu       = torch.zeros(n_samples, n_t, n_loc)

    with torch.no_grad():
        for i in range(n_samples):
            y_star, _, _, _, _ = model(full_data.x.to(device), full_data.y.to(device))
            y_star_t = y_star[:, n_loc:2 * n_loc]
            err1 = torch.empty_like(y_star_t).exponential_(ALPHA0)
            err2 = torch.empty_like(y_star_t).bernoulli_(0.5) * 2 - 1
            Emu[i] = err1.mul(err2).exp().mul(y_star_t).cpu()
            if (i + 1) % 100 == 0:
                print(f"Emulated {i + 1}/{n_samples}")

    return Emu


def back_transform(emu_samples, gev_par):
    """
    Apply inverse GEV transformation to recover original FWI scale.

    Args:
        emu_samples: Tensor (n_samples x n_t x n_loc)
        gev_par:     Tensor (n_loc x 3) — location, scale, shape

    Returns:
        Tensor (n_samples x n_t x n_loc) on original FWI scale
    """
    gev_beta = gev_par[:, 0] - gev_par[:, 1] / gev_par[:, 2]
    result   = torch.zeros_like(emu_samples)

    for i in range(gev_par.shape[0]):
        xi  = gev_par[i, 2].item()
        tau = gev_par[i, 1].item()
        b   = gev_beta[i].item()
        tmp = emu_samples[:, :, i]

        if xi > 0:
            result[:, :, i] = tmp.pow(xi).mul(tau).div(xi).add(b)
        elif xi < 0:
            result[:, :, i] = tmp.pow(-xi).mul(xi).pow(-1).mul(tau).add(b)

    return result


# Generate factual and counterfactual emulations
print("Generating factual emulations...")
Emu_factual = emulate(model, Nina34_input, n_samples=500)
Emu_factual_ori = back_transform(Emu_factual, GEV_par)

fake_Nina34 = MEI_SCALE - Nina34_input
print("Generating counterfactual emulations...")
Emu_counterfact = emulate(model, fake_Nina34, n_samples=500)
Emu_counterfact_ori = back_transform(Emu_counterfact, GEV_par)

torch.save(Emu_factual_ori,    os.path.join(DATA_DIR, "Emu_factual_ori.pt"))
torch.save(Emu_counterfact_ori, os.path.join(DATA_DIR, "Emu_counterfact_ori.pt"))
print("Emulations saved.")

# =============================================================================
# Spatial location tensor for evaluation
# =============================================================================

lon_vals = torch.linspace(LON_MIN, LON_MAX, 26)
lat_vals = torch.linspace(LAT_MIN, LAT_MAX, 43)
lon_grid, lat_grid = torch.meshgrid(lon_vals, lat_vals, indexing="xy")
location_tensor = torch.stack((lon_grid.flatten(), lat_grid.flatten()), dim=-1)

X_train_true = torch.tensor(X[:, :])

# =============================================================================
# Chi-Coefficient Comparison
# =============================================================================

u_vec        = torch.cat([torch.arange(0.90, 0.98, 0.001),
                           torch.arange(0.981, 0.999, 0.0001)])
distance_vec = torch.tensor([2.0, 6.0, 10.0])
N_REPS       = 10

comparisons = {}
for com_ind, distance in enumerate(distance_vec):
    d = float(distance)
    chi_truth = utils.chi_est(
        Data=X_train_true, Loc=location_tensor,
        d=d, tol=0.1, gridded=True, u_vec=u_vec
    )
    chi_reps = torch.zeros(N_REPS, 3, len(u_vec))

    for rep in range(N_REPS):
        print(f"  Chi rep {rep + 1}/{N_REPS}, distance={d}")
        model.eval()
        with torch.no_grad():
            y_star, _, _, _, _ = model(full_dataset.x, full_dataset.y)
        y_star_t = y_star[:, n_loc:2 * n_loc]
        err1 = torch.empty_like(y_star_t).exponential_(ALPHA0)
        err2 = torch.empty_like(y_star_t).bernoulli_(0.5) * 2 - 1
        X_emu = err1.mul(err2).exp().mul(y_star_t).to(torch.float64)

        tmp = utils.chi_est(
            Data=X_emu.T, Loc=location_tensor,
            d=d, tol=0.1, gridded=True, u_vec=u_vec
        )
        chi_reps[rep, 0] = torch.tensor(tmp["truth"])
        chi_reps[rep, 1] = torch.tensor(tmp["upper"])
        chi_reps[rep, 2] = torch.tensor(tmp["lower"])

    comparisons[com_ind] = {
        **chi_truth,
        "emu":       chi_reps[:, 0].mean(0),
        "emu_upper": chi_reps[:, 1].mean(0),
        "emu_lower": chi_reps[:, 2].mean(0),
    }

fig = plt.figure(figsize=(8, 6))
linestyles = ["-", "--", ":"]
for i, (_, res) in enumerate(sorted(comparisons.items())):
    ls = linestyles[i % 3]
    plt.plot(res["u"], res["truth"],       color="r", linestyle=ls, linewidth=1.5)
    plt.fill_between(res["u"], res["lower"], res["upper"], color="red",  alpha=0.1)
    plt.plot(res["u"], res["emu"].numpy(), color="b", linestyle=ls, linewidth=1.5)
    plt.fill_between(res["u"], res["emu_lower"].numpy(),
                     res["emu_upper"].numpy(), color="blue", alpha=0.1)

for label, ypos in zip(["Short-range", "Medium-range", "Long-range"], [0.62, 0.48, 0.27]):
    plt.text(0.15, ypos, label, fontsize=12, transform=plt.gca().transAxes)

plt.ylim(0, 1)
plt.xlabel("Quantile (u)", fontsize=16)
plt.ylabel(r"$\chi(u)$", fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "chi_FWI.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Chi plot saved.")

# =============================================================================
# ARE Comparison
# =============================================================================

u_vec_are = torch.cat([torch.arange(0, 0.98, 0.005),
                        torch.arange(0.980, 0.999, 0.001)])
unit_area  = 0.078125   # grid cell area in degrees^2

ARE_truth = torch.zeros(3, len(u_vec_are))
for i, u in enumerate(u_vec_are):
    ARE_truth[:, i] = utils.ARE(u, unit_area=unit_area, x=X_train_true)

ARE_res = torch.zeros(N_REPS, 3, len(u_vec_are))
Emu_Data = torch.zeros(N_REPS, n_loc, n_t)

for rep in range(N_REPS):
    print(f"ARE rep {rep + 1}/{N_REPS}")
    model.eval()
    with torch.no_grad():
        y_star, _, _, _, _ = model(full_dataset.x, full_dataset.y)
    y_star_t = y_star[:, n_loc:2 * n_loc]
    err1 = torch.empty_like(y_star_t).exponential_(ALPHA0)
    err2 = torch.empty_like(y_star_t).bernoulli_(0.5) * 2 - 1
    Emu_Data[rep] = err1.mul(err2).exp().mul(y_star_t).T.cpu()
    for i, u in enumerate(u_vec_are):
        ARE_res[rep, :, i] = utils.ARE(u, unit_area=unit_area, x=Emu_Data[rep])

ARE_mean = ARE_res[:, 0].mean(0)
ARE_low  = ARE_res[:, 1].mean(0)
ARE_high = ARE_res[:, 2].mean(0)

plt.figure(figsize=(8, 5))
plt.plot(u_vec_are, ARE_truth[0], "r-", label="Truth")
plt.fill_between(u_vec_are, ARE_truth[1], ARE_truth[2],
                 color="tomato", alpha=0.5, label="95% CI (Truth)")
plt.plot(u_vec_are, ARE_mean, "b-", label="Emulation")
plt.fill_between(u_vec_are, ARE_low, ARE_high,
                 color="lightblue", alpha=0.5, label="95% CI (Emulation)")
plt.xlabel("Quantile", fontsize=18)
plt.ylabel("Averaged Radius of Exceedances", fontsize=18)
plt.tick_params(labelsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ARE_FWI.png"), dpi=300, bbox_inches="tight")
plt.close()
print("ARE plot saved.")

# =============================================================================
# Tail-Weighted CRPS
# =============================================================================

def compute_tail_weighted_crps(X_target, emulation, tail_quantile=0.9):
    """
    Args:
        X_target:  Tensor (n_holdout x n_time)
        emulation: Tensor (n_holdout x n_time x n_sample)
    Returns:
        CRPS: Tensor (n_holdout x n_time)
    """
    n_loc_, n_time_, n_sample = emulation.shape
    CRPS = torch.zeros(n_loc_, n_time_)
    for loc in range(n_loc_):
        for t in range(n_time_):
            samples  = torch.sort(emulation[loc, t])[0]
            z_tail   = samples[int(n_sample * tail_quantile)]
            upper    = torch.max(torch.cat([samples, X_target[loc, t].unsqueeze(0)])) + 1e-4
            z_grid   = torch.linspace(0.0, upper.item(), 1000)
            ecdf     = torch.tensor([torch.mean((samples <= z).float()) for z in z_grid])
            indicator = (X_target[loc, t] <= z_grid).float()
            weight    = (z_grid > z_tail).float()
            CRPS[loc, t] = torch.trapz(weight * (ecdf - indicator) ** 2,
                                        dx=z_grid[1] - z_grid[0])
        if (loc + 1) % 2 == 0:
            print(f"  CRPS location {loc + 1}/{n_loc_}")
    return CRPS


n_crps   = 2000
cand_ind = range(n_t)

white_noise = torch.rand_like(Nina34_input) * MEI_SCALE
fake_y      = MEI_SCALE - full_dataset.y

Emu_holdout       = torch.zeros(n_holdout, n_t, n_crps)
Emu_holdout_white = torch.zeros_like(Emu_holdout)

for i in range(n_crps):
    if (i + 1) % 200 == 0:
        print(f"CRPS sample {i + 1}/{n_crps}")
    model.eval()
    with torch.no_grad():
        y_star, _, _, _, _ = model(full_dataset.x, full_dataset.y)
        y_wn, _, _, _, _   = model(full_dataset.x, white_noise)
    y_t = y_star[:, n_loc:2 * n_loc]
    e1  = torch.empty_like(y_t).exponential_(ALPHA0)
    e2  = torch.empty_like(y_t).bernoulli_(0.5) * 2 - 1
    Emu_holdout[:, :, i]       = (e1 * e2).exp().mul(y_t)[:, holdout_idx].T.cpu()

    y_wn_t = y_wn[:, n_loc:2 * n_loc]
    e1 = torch.empty_like(y_wn_t).exponential_(ALPHA0)
    e2 = torch.empty_like(y_wn_t).bernoulli_(0.5) * 2 - 1
    Emu_holdout_white[:, :, i] = (e1 * e2).exp().mul(y_wn_t)[:, holdout_idx].T.cpu()

CRPS_model = compute_tail_weighted_crps(X_holdout, Emu_holdout)
CRPS_white = compute_tail_weighted_crps(X_holdout, Emu_holdout_white)

fig, ax = plt.subplots(figsize=(7, 5))
data  = [CRPS_model.log().view(-1).numpy(), CRPS_white.log().view(-1).numpy()]
parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("skyblue"); pc.set_edgecolor("black"); pc.set_alpha(0.8)
parts["cmedians"].set_color("darkblue"); parts["cmedians"].set_linewidth(1.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["cXVAE", "cXVAE*"], fontsize=16)
ax.tick_params(labelsize=16)
ax.set_ylabel("Tail-weighted CRPS (log scale)", fontsize=18)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "crps_FWI.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print("CRPS violin plot saved.")

# =============================================================================
# Q-Q Plot
# =============================================================================

def qq_plot(x, y, lim=50, fontsize=18, xlabel="Truth",
            ylabel="Emulation", save_path=None):
    mask = ~torch.isnan(x) & ~torch.isnan(y)
    x, y = x[mask], y[mask]
    p    = torch.linspace(1e-4, 1 - 1e-4, 200)
    xq   = torch.quantile(x, p)
    yq   = torch.quantile(y, p)
    K    = 1.36
    M    = len(x) * len(y) / (len(x) + len(y))
    yl   = torch.quantile(y, torch.clamp(p - K / np.sqrt(M), 0, 1))
    yu   = torch.quantile(y, torch.clamp(p + K / np.sqrt(M), 0, 1))
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
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


qq_idx = 2
qq_plot(X_holdout[qq_idx].to(torch.float32),
        Emu_holdout[qq_idx, :, 0].to(torch.float32),
        lim=50, ylabel="cXVAE Emulation",
        save_path=os.path.join(FIG_DIR, "qq_FWI.pdf"))
qq_plot(X_holdout[qq_idx].to(torch.float32),
        Emu_holdout_white[qq_idx, :, 0].to(torch.float32),
        lim=50, ylabel="cXVAE* Emulation",
        save_path=os.path.join(FIG_DIR, "qq_FWI_white.pdf"))
print("Q-Q plots saved.")

# =============================================================================
# Counterfactual Kernel Density Plots
# =============================================================================

months     = pd.date_range(start="2014-05-01", end="2024-11-01", freq="MS")
month_labels = months.strftime("%Y-%m")

# Select representative time points: Nov 2023, Apr 2024, Oct 2024
target_times = [114, 119, 125]   # 0-indexed

for t_idx in target_times:
    s1_f  = Emu_factual_ori[:, t_idx, 22 * 26 + 16].numpy()
    s2_f  = Emu_factual_ori[:, t_idx, 18 * 26 + 12].numpy()
    s1_cf = Emu_counterfact_ori[:, t_idx, 22 * 26 + 16].numpy()
    s2_cf = Emu_counterfact_ori[:, t_idx, 18 * 26 + 12].numpy()

    def kde_grid(a, b, n=100):
        x = np.linspace(float(min(a)), float(max(a)), n)
        y = np.linspace(float(min(b)), float(max(b)), n)
        X_, Y_ = np.meshgrid(x, y)
        Z = np.reshape(
            gaussian_kde(np.vstack([a, b]), bw_method=0.8)(
                np.vstack([X_.ravel(), Y_.ravel()])
            ).T, X_.shape
        )
        return X_, Y_, Z / Z.max()

    X1, Y1, Z1 = kde_grid(s1_f,  s2_f)
    X2, Y2, Z2 = kde_grid(s1_cf, s2_cf)
    levels = np.linspace(0.1, 1, 10)

    fig, ax = plt.subplots(figsize=(8, 8))
    c1 = ax.contour(X1, Y1, Z1, levels=levels, colors="red",  linewidths=1)
    c2 = ax.contour(X2, Y2, Z2, levels=levels, colors="blue", linewidths=1)
    plt.clabel(c1, inline=True, fontsize=10, fmt="%.1f")
    plt.clabel(c2, inline=True, fontsize=10, fmt="%.1f")
    ax.legend(handles=[
        Line2D([0], [0], color="red",  lw=1, label="cXVAE Emulation"),
        Line2D([0], [0], color="blue", lw=1, label="Counterfactual")
    ], fontsize=18)
    ax.set_xlabel(r"$X(\mathbf{s}_1)$", fontsize=22)
    ax.set_ylabel(r"$X(\mathbf{s}_2)$", fontsize=22)
    ax.tick_params(labelsize=17)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, f"kernel_FWI_{month_labels[t_idx]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Kernel density plot saved: {save_path}")

# =============================================================================
# Spatial Field Comparison (theta and log(X))
# =============================================================================

model.eval()
with torch.no_grad():
    y_star, random_coefficients, _, _, _ = model(full_dataset.x, full_dataset.y)

y_star_t  = y_star[:, n_loc:2 * n_loc]
err1      = torch.empty_like(y_star_t).exponential_(ALPHA0)
err2      = torch.empty_like(y_star_t).bernoulli_(0.5) * 2 - 1
X_emu     = err1.mul(err2).exp().mul(y_star_t)

theta_pred = (random_coefficients @ rbf_mat.T).relu()

# Plot theta and log(X) at 3 representative times
fig, axes = plt.subplots(3, 3, figsize=(10, 11),
                          subplot_kw={"projection": ccrs.PlateCarree()})
col_titles = ["November 2023", "April 2024", "October 2024"]
row_labels = [r"$\theta$ (Emulated)", r"$\log(X)$ (True)", r"$\log(X)$ (Emulated)"]

lon = np.linspace(LON_MIN, LON_MAX, 26)
lat = np.linspace(LAT_MIN, LAT_MAX, 43)
lon_m, lat_m = np.meshgrid(lon, lat)

im_theta = im_field = None

for row in range(3):
    for col, t_idx in enumerate(target_times):
        ax = axes[row, col]
        if row == 0:
            data = theta_pred[t_idx].reshape(43, 20).detach().numpy()
            im   = ax.pcolormesh(lon_m[:, :20], lat_m[:, :20], data,
                                 cmap="Reds", vmin=0, vmax=0.045,
                                 transform=ccrs.PlateCarree())
            if col == 0: im_theta = im
        else:
            src  = X_train_true if row == 1 else X_emu.T
            data = src[:, t_idx].log().reshape(43, 26).detach().numpy()
            im   = ax.pcolormesh(lon_m, lat_m, data,
                                 cmap="inferno", vmin=-3.0, vmax=4.0,
                                 transform=ccrs.PlateCarree())
            if row == 1 and col == 0: im_field = im

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=14)

for label, ypos in zip(row_labels, [0.78, 0.48, 0.21]):
    fig.text(0.015, ypos, label, va="center", ha="left", fontsize=13, rotation=90)

fig.add_axes([0.89, 0.65, 0.015, 0.25])
fig.colorbar(im_theta, cax=fig.axes[-1]).set_label(r"$\theta$", fontsize=14)
fig.add_axes([0.89, 0.1,  0.015, 0.35])
fig.colorbar(im_field, cax=fig.axes[-1]).set_label(r"$\log(X)$", fontsize=14)

plt.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.05,
                    wspace=0.05, hspace=0.2)
plt.savefig(os.path.join(FIG_DIR, "spatial_FWI.png"), dpi=300)
plt.close()
print("Spatial field comparison saved.")

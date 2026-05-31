# =============================================================================
# train_cXVAE.py
#
# Define and train the conditional XVAE (cXVAE) model for spatio-temporal
# extremes, conditioned on the ENSO climate index.
#
# Prerequisites:
#   - simulate_data.R has been run and CSV outputs are in DATA_DIR
#   - pretrain_CNN.py has been run and CNN_pretrained.pt exists (recommended)
#
# Output:
#   cXVAE_trained.pt  — saved cXVAE model weights (best validation epoch)
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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import utils

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR        = "data/simulation"
PRETRAINED_CNN  = "CNN_pretrained.pt"
MODEL_OUT       = "cXVAE_trained.pt"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ALPHA         = 0.5
MEI_SCALE     = 2.0
LEARNING_RATE = 1e-8
NUM_EPOCHS    = 200
BATCH_SIZE    = 32
VAL_RATIO     = 0.1
PATIENCE      = 20
MIN_DELTA     = 1e-4    # relative improvement threshold for early stopping

# =============================================================================
# Load Data
# =============================================================================

def load_csv(filename):
    return pd.read_csv(
        os.path.join(DATA_DIR, filename),
        header=None, skiprows=1
    ).drop([0], axis="columns").values


W       = load_csv("W_Data.csv")
X       = load_csv("X_Data.csv")
Z       = load_csv("Z_Data.csv")
RBF     = load_csv("RBF_Data.csv")
MEIs_MA = load_csv("MEIs_MA_Data.csv")
y       = load_csv("Y_Data.csv")

k          = W.shape[1]         # number of knots (225)
knot_width = int(k ** 0.5)      # 15
alpha_size = RBF.shape[1]       # 144
n_loc      = X.shape[0]         # 2500
n_t        = X.shape[1]         # 528
latent_size = k * 3
hidden_size = k * 3
image_size  = n_loc

MEIs_MA = MEIs_MA * MEI_SCALE

# Convert to tensors
W_alpha    = torch.tensor(W).pow(1 / ALPHA).to(torch.float32).to(DEVICE)
X_tensor   = torch.tensor(X).to(torch.float32).to(DEVICE)
MEI_tensor = torch.tensor(MEIs_MA).T.to(DEVICE)
rbf_mat    = torch.tensor(RBF).to(torch.float32).to(DEVICE)
Y_tensor   = torch.tensor(y).to(torch.float32).to(DEVICE)

# =============================================================================
# Holdout Location Selection (extreme-rich locations)
# =============================================================================

q         = 0.9
threshold = torch.quantile(X_tensor, q)
count_high = (X_tensor > threshold).sum(dim=1)
_, sorted_idx = torch.sort(count_high)

n_holdout  = 10
holdout_idx = sorted_idx[n_loc - n_holdout:]
train_idx   = np.setdiff1d(np.arange(n_loc), holdout_idx.cpu().numpy())

X_train   = X_tensor
X_holdout = X_tensor[holdout_idx, :]
W_train   = W_alpha

# =============================================================================
# Encoder Initialization via Truncated SVD
# =============================================================================

U, D, Vt = np.linalg.svd(W_alpha.cpu().numpy(), full_matrices=False)
keep    = D > (1e-2 * np.max(D))
U_k, D_k, V_k = U[:, keep], D[keep], Vt[keep, :].T
proj    = torch.tensor(V_k @ np.diag(1 / D_k) @ U_k.T).T.to(torch.float32).to(DEVICE)

# Build block-diagonal projection and W_alpha matrices for encoder init
def build_block_diag(M):
    r, c = M.shape
    z = torch.zeros(r, c, device=M.device)
    return torch.cat([
        torch.cat([M, z, z]),
        torch.cat([z, M, z]),
        torch.cat([z, z, M])
    ], dim=1)

proj_final    = build_block_diag(proj.T).to(DEVICE)
W_alpha_final = build_block_diag(W_train.T).to(DEVICE)

# =============================================================================
# Dataset and DataLoader
# =============================================================================

X_train_tensor = utils.x_aug(X_train)
MEI_input      = utils.x_aug(MEI_tensor)

full_dataset  = utils.CVAEinput_Dataset(X_train_tensor, MEI_input)
n_val         = int(VAL_RATIO * len(full_dataset))
n_train       = len(full_dataset) - n_val

train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# CNN Submodule (same architecture as pretrain_CNN.py)
# =============================================================================

class CNN(nn.Module):
    def __init__(self, alpha_size, input_shape=(3, 15, 30)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,  3, padding="same")
        self.conv2 = nn.Conv2d(32, 64,  3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv6 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv7 = nn.Conv2d(64, 32,  3, padding="same")
        self.conv8 = nn.Conv2d(32, 16,  3, padding="same")

        with torch.no_grad():
            dummy    = torch.zeros(1, *input_shape)
            flat_dim = self._forward_conv(dummy).view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, alpha_size)

    def _forward_conv(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv6(x); x = self.conv7(x); x = self.conv8(x)
        return x

    def forward(self, x):
        x = torch.flatten(self._forward_conv(x), start_dim=1)
        return self.fc1(x).view(-1, alpha_size)

# =============================================================================
# cXVAE Model
# =============================================================================

class CVAE(nn.Module):
    """
    Conditional XVAE (cXVAE) for spatio-temporal extremes.

    Architecture:
      Encoder:     MLP with Softplus activation; maps X_t -> (mu_t, sigma_t)
      Latent space: Log-scale reparameterization with linear ENSO injection g(c_t)
      CNN Decoder: Fused (Z_t, c_t) -> CNN -> basis coefficients xi_t -> theta_t
                   Learnable W maps Z_t -> Y_t; log-Laplace noise gives X_t
    """
    def __init__(self, alpha_size):
        super().__init__()

        # Encoder
        self.fc1      = nn.Linear(image_size, hidden_size)
        self.fc2      = nn.Linear(hidden_size, latent_size)
        self.fc3      = nn.Linear(hidden_size, latent_size)
        self.softplus = nn.Softplus(beta=8)

        # CNN decoder (for theta estimation)
        self.conv1 = nn.Conv2d(3, 32,  3, padding="same")
        self.conv2 = nn.Conv2d(32, 64,  3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv6 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv7 = nn.Conv2d(64, 32,  3, padding="same")
        self.conv8 = nn.Conv2d(32, 16,  3, padding="same")
        self.fct   = nn.Linear(7200, alpha_size)

        # ENSO linear injection: g(c_t) = b * c_t
        self.reg_lin = nn.Linear(3, k * 3, bias=False)

        # Spatial decoder W: learnable basis mapping Z_t -> Y_t
        self.l1 = nn.Linear(latent_size * 2, image_size)

    def encode(self, x):
        h1      = self.softplus(self.fc1(x)) + 1e-5
        mu      = self.fc2(h1)
        log_var = self.fc3(h1)
        return mu, torch.exp(log_var)

    def reparameterize(self, mu, var, label):
        eps      = torch.randn(var.shape, device=var.device)
        reg_term = self.reg_lin(label)
        z        = mu.log() + torch.sqrt(var) * eps + reg_term
        return z, eps

    def decode(self, z, label):
        # Build CNN input: interleave z and label
        z_inputs = utils.z_chess_aug_2k(z, label, width=knot_width).to(self.conv1.weight.device)

        # CNN: estimate basis coefficients for theta
        t1 = self.conv2(self.conv1(z_inputs))
        t1 = self.conv6(self.conv3(t1))
        t1 = self.conv8(self.conv7(t1))
        random_coefficients = self.fct(torch.flatten(t1, start_dim=1)).view(-1, alpha_size)

        # Spatial decoder: W * z -> Y_t
        z_origin = z_inputs.exp().reshape(z_inputs.shape[0], -1)
        z_odd    = torch.log(z_origin[:, 1::2])
        z_even   = z_origin[:, 0::2]
        z_output = torch.stack((z_even, z_odd), dim=2).reshape(z_origin.shape)
        y_star   = self.softplus(self.l1(z_output))

        return y_star, random_coefficients

    def forward(self, x, label):
        mu, var                    = self.encode(x)
        z, eps                     = self.reparameterize(mu, var, label)
        y_star, random_coefficients = self.decode(z, label)
        return y_star, random_coefficients, eps, var, z

# =============================================================================
# Model Initialization
# =============================================================================

torch.manual_seed(12)
model = CVAE(alpha_size=alpha_size).to(DEVICE)

# Initialize decoder W with Wendland basis (block-diagonal structure)
mask = torch.clone(model.l1.weight)
mask[:, 0::2] = W_alpha_final.T
model.l1.weight = nn.Parameter(mask)
model.l1.bias   = nn.Parameter(torch.zeros_like(model.l1.bias))

# Initialize encoder with SVD-based projection
model.fc1.weight = nn.Parameter(proj_final)
model.fc1.bias   = nn.Parameter(torch.zeros_like(model.fc1.bias))
model.fc2.weight = nn.Parameter(torch.diag(torch.ones(model.fc2.weight.shape[0])))
model.fc2.bias   = nn.Parameter(torch.zeros_like(model.fc2.bias))
model.fc3.weight = nn.Parameter(torch.zeros_like(model.fc3.weight))
model.fc3.bias   = nn.Parameter(torch.full_like(model.fc3.bias, -10.0))
model.reg_lin.weight = nn.Parameter(torch.zeros_like(model.reg_lin.weight))

# Load pretrained CNN weights into decoder (recommended)
if os.path.exists(PRETRAINED_CNN):
    pretrained = CNN(alpha_size=alpha_size).to(DEVICE)
    pretrained.load_state_dict(torch.load(PRETRAINED_CNN, map_location=DEVICE))
    for layer in ["conv1", "conv2", "conv3", "conv6", "conv7", "conv8"]:
        getattr(model, layer).load_state_dict(getattr(pretrained, layer).state_dict())
    model.fct.load_state_dict(pretrained.fc1.state_dict())
    print(f"Loaded pretrained CNN weights from {PRETRAINED_CNN}")
else:
    print(f"Warning: {PRETRAINED_CNN} not found. Training from scratch (not recommended).")

# =============================================================================
# Loss Function (ELBO)
# =============================================================================

ALPHA0 = 30.0  # log-Laplace noise scale (matches simulation)

def loss_function(x, label, var, eps, y_star, rbf_mat, random_coefficients, z):
    """
    Negative ELBO for the cXVAE:
      loss_p1: log-Laplace data likelihood
      loss_p2: expPS prior on latent variables
      loss_p3: KL divergence (encoder vs prior)
      loss_p4: temporal smoothness penalty on xi_t
    """
    # Part 1: log-Laplace likelihood
    standardized = x.div(y_star + 1e-4)
    loss_p1 = (standardized.log().abs().mul(-ALPHA0) - standardized.log())
    loss_p1 = loss_p1.sum(dim=1).mean().div(3)

    # Part 2: expPS log-likelihood on latent z
    theta       = torch.mm(rbf_mat, random_coefficients.T).relu()
    z_reshaped  = z.reshape(-1, k).T
    theta_expand = theta.repeat(1, 3)
    log_lik_z   = (theta_expand.sqrt()
                   - z_reshaped.mul(1.5)
                   - z_reshaped.exp().mul(theta_expand)
                   - z_reshaped.exp().mul(4).pow(-1))
    loss_p2 = log_lik_z.sum(dim=0).mean()

    # Part 3: KL divergence
    loss_p3 = (var.log().sum(dim=1) + eps.pow(2).mul(0.5).sum(dim=1)).mean().div(3)

    # Part 4: temporal smoothness on basis coefficients
    d1     = torch.diff(random_coefficients, dim=0)
    ts     = torch.diff(label, dim=1).abs()
    ts_mid = ts[:, 0][:-1].clamp(min=0.01)
    loss_p4 = 0.05 * d1.div(ts_mid.unsqueeze(1)).pow(2).sum().sqrt().div(d1.numel())

    return loss_p1, loss_p2, loss_p3, loss_p4

# =============================================================================
# Optimizer (layer-specific learning rates)
# =============================================================================

optimizer = optim.Adam([
    {"params": [p for n, p in model.named_parameters()
                if n not in ["l1.weight", "reg_lin.weight",
                             "fc1.weight", "fc2.weight", "fc3.weight"]],
     "lr": 1e-10},                 # CNN layers — nearly frozen
    {"params": [model.fc1.weight, model.fc2.weight, model.fc3.weight],
     "lr": 1e-9},                  # encoder — small
    {"params": [model.l1.weight],
     "lr": 1e-6},                  # spatial decoder W — main learnable component
    {"params": [model.reg_lin.weight],
     "lr": 1e-9},                  # ENSO injection weight
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-10
)

# =============================================================================
# Training Loop with Early Stopping
# =============================================================================

best_val_loss   = float("inf")
best_epoch      = 0
epochs_no_improve = 0
best_model_state  = None
loss_list = []

start_time = time.time()
print(f"Training cXVAE on {DEVICE}...")

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
        loss = (loss_p1 + loss_p2 + loss_p3).mul(-1) + loss_p4
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
            val_loss += (loss_p1 + loss_p2 + loss_p3).mul(-1).item() + loss_p4.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    loss_list.append(train_loss)

    print(f"Epoch {epoch + 1}: Train = {train_loss:.4f} | Val = {val_loss:.4f}")

    # Early stopping
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

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best model from epoch {best_epoch}.")

elapsed = time.time() - start_time
print(f"Training complete in {elapsed:.2f} seconds.")

torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# =============================================================================
# Emulation Function
# =============================================================================

def cXVAE_emulate(model, n_samples, condition_data, n_loc=2500, alpha0=30.0):
    """
    Generate emulated spatial fields by passing observed data and condition
    through the trained model.

    The encoder maps the observed spatial fields to the latent space, the
    condition c is injected via the learned linear mapping g(c), and new
    spatial fields are generated by introducing fresh log-Laplace noise.
    Counterfactual emulations are obtained by substituting a different
    condition_data (e.g. flipped ENSO) while keeping the observed fields fixed.

    Args:
        model:          trained CVAE model
        n_samples:      int — number of emulated replicates
        condition_data: Tensor (n_t x 3) — ENSO condition inputs [t-1, t, t+1]
        n_loc:          int — number of spatial locations
        alpha0:         float — log-Laplace noise parameter

    Returns:
        emulated_results: Tensor (n_loc x n_t x n_samples)
    """
    model.eval()
    device     = next(model.parameters()).device
    cond_input = condition_data.to(device)
    n_t_emu    = cond_input.shape[0]

    X_input   = utils.x_aug(X_tensor).to(device)
    full_data = utils.CVAEinput_Dataset(X_input, cond_input)
    emulated  = torch.zeros(n_loc, n_t_emu, n_samples)

    with torch.no_grad():
        for i in range(n_samples):
            y_star, _, _, _, _ = model(full_data.x.to(device), full_data.y.to(device))
            y_star_t = y_star[:, n_loc:2 * n_loc]   # current time slice

            err1  = torch.empty_like(y_star_t).exponential_(alpha0)
            err2  = torch.empty_like(y_star_t).bernoulli_(0.5) * 2 - 1
            X_emu = err1.mul(err2).exp().mul(y_star_t)

            emulated[:, :, i] = X_emu.T.cpu()

            if (i + 1) % 500 == 0:
                print(f"Emulated {i + 1}/{n_samples}")

    return emulated

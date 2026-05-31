# =============================================================================
# baseline_SohnCVAE.py
#
# Vanilla conditional VAE baseline (Sohn et al., 2015) for comparison
# with the cXVAE model.
#
# The decoder's final layer is initialized with the Wendland basis matrix
# to ensure a fair spatial comparison with the cXVAE — both models start
# with the same basis expansion setup. Any remaining performance gap
# therefore reflects genuine architectural differences.
#
# Output:
#   SohnCVAE_trained.pt — saved model weights
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import utils

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR  = "data/simulation"
MODEL_OUT = "SohnCVAE_trained.pt"

LATENT_DIM = 16
N_BASIS    = 225
ALPHA      = 0.5
MEI_SCALE  = 2.0
EPOCHS     = 500
PATIENCE   = 30
BATCH_SIZE = 32

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
MEIs_MA = load_csv("MEIs_MA_Data.csv") * MEI_SCALE

n_loc = X.shape[0]   # 2500

# Wendland basis: W_alpha shape (n_loc x k) -> used for decoder init
W_tensor = torch.tensor(W).to(torch.float32)
W_alpha  = W_tensor.pow(1 / ALPHA)   # (2500, 225) — Wendland^(1/alpha)

# Scale W_alpha_basis to match data range (ensures decoder output is in correct range)
# Note: W_alpha_basis is distinct from W_alpha used in the cXVAE encoder
X_tensor = torch.tensor(X).to(torch.float32)   # (2500, 528)

target_mean              = X_tensor.mean().item()
current_basis_contribution = W_alpha.mean().item() * N_BASIS
basis_scale              = target_mean / current_basis_contribution
W_alpha_basis            = W_alpha * basis_scale   # (2500, 225) — scaled for decoder init

# =============================================================================
# Dataset
# =============================================================================

# y_tensor: (n_t x 3 x n_loc) — three time-channel pseudo-replicates
# x_tensor: (n_t x 3)          — ENSO condition [t-1, t, t+1]

MEI_tensor = torch.tensor(MEIs_MA).to(torch.float32)
X_full     = utils.x_aug(X_tensor.T.unsqueeze(0).squeeze(0))  # not used here directly

y_tensor = X_tensor.T.unsqueeze(1).expand(-1, 3, -1)  # (528, 3, 2500)

# Build [c_{t-1}, c_t, c_{t+1}] condition vector for each time
n_t = MEI_tensor.shape[0]
x_cond = torch.zeros(n_t, 3)
for t in range(n_t):
    t_prev = t if t == 0 else t - 1
    t_next = t if t == n_t - 1 else t + 1
    x_cond[t] = torch.tensor([MEI_tensor[t_prev, 0],
                               MEI_tensor[t, 0],
                               MEI_tensor[t_next, 0]])

dataset   = TensorDataset(y_tensor, x_cond)
n_val     = 100
n_train   = n_t - n_val
train_set, val_set = random_split(dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

# =============================================================================
# Model Definition
# =============================================================================

class SohnCVAE_Wendland(nn.Module):
    """
    Vanilla conditional VAE (Sohn et al., 2015) with Wendland basis
    initialization in the decoder for fair spatial comparison with cXVAE.

    The encoder uses 1D convolutions over the 2500-location spatial field.
    The decoder predicts N_BASIS coefficients and reconstructs the field
    via a linear layer initialized with the Wendland basis matrix.
    """
    def __init__(self, x_dim, W_alpha_basis, latent_dim=LATENT_DIM, n_basis=N_BASIS):
        super().__init__()
        self.n_basis = n_basis

        # Encoder: q(z | y, x)
        self.enc_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=15, stride=5, padding=7),  # 2500 -> 500
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=5, padding=7), # 500  -> 100
            nn.ReLU(),
            nn.Flatten()
        )
        self.enc_fc   = nn.Linear(64 * 100 + x_dim, 256)
        self.mu_q     = nn.Linear(256, latent_dim)
        self.logvar_q = nn.Linear(256, latent_dim)

        # Conditional prior: p(z | x)
        self.prior_net = nn.Sequential(
            nn.Linear(x_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU()
        )
        self.mu_p     = nn.Linear(128, latent_dim)
        self.logvar_p = nn.Linear(128, latent_dim)

        # Decoder: (z, x) -> n_basis coefficients -> 2500 field
        self.dec_fc1 = nn.Linear(latent_dim + x_dim, 128)
        self.dec_fc2 = nn.Linear(128, n_basis)

        # One Wendland-initialized reconstruction layer per time channel
        self.l1_tm1 = nn.Linear(n_basis, 2500)
        self.l1_t0  = nn.Linear(n_basis, 2500)
        self.l1_tp1 = nn.Linear(n_basis, 2500)

        # Initialize with scaled Wendland basis
        for layer in [self.l1_tm1, self.l1_t0, self.l1_tp1]:
            with torch.no_grad():
                layer.weight.copy_(W_alpha_basis)   # (2500, 225)
                nn.init.zeros_(layer.bias)

        self.softplus = nn.Softplus(beta=8)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, x):
        h     = F.relu(self.dec_fc1(torch.cat([z, x], dim=1)))
        h     = self.dec_fc2(h)                         # (B, n_basis)
        y_tm1 = self.softplus(self.l1_tm1(h))           # (B, 2500)
        y_t0  = self.softplus(self.l1_t0(h))
        y_tp1 = self.softplus(self.l1_tp1(h))
        return torch.stack([y_tm1, y_t0, y_tp1], dim=1) # (B, 3, 2500)

    def forward(self, y, x):
        # Prior p(z | x)
        h_p               = self.prior_net(x)
        mu_p, logvar_p    = self.mu_p(h_p), self.logvar_p(h_p)

        # Encoder q(z | y, x)
        h_q               = self.enc_conv(y)
        h_q               = F.relu(self.enc_fc(torch.cat([h_q, x], dim=1)))
        mu_q, logvar_q    = self.mu_q(h_q), self.logvar_q(h_q)

        # Sample and decode
        z     = self.reparameterize(mu_q, logvar_q)
        y_hat = self.decode(z, x)

        return y_hat, mu_q, logvar_q, mu_p, logvar_p

# =============================================================================
# Loss Function
# =============================================================================

def cvae_loss(y_hat, y, mu_q, logvar_q, mu_p, logvar_p, kl_weight=1.0):
    """MSE reconstruction loss + KL divergence KL(q || p)."""
    recon_loss = F.mse_loss(y_hat, y, reduction="sum")
    var_q      = torch.exp(logvar_q)
    var_p      = torch.exp(logvar_p)
    kl_div     = 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1
    )
    return recon_loss + kl_weight * kl_div

# =============================================================================
# Training
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for y_batch, x_batch in train_loader:
            optimizer.zero_grad()
            y_hat, mu_q, logvar_q, mu_p, logvar_p = model(y_batch, x_batch)
            loss = cvae_loss(y_hat, y_batch, mu_q, logvar_q, mu_p, logvar_p)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_val, x_val in val_loader:
                y_hat_v, mu_q_v, logvar_q_v, mu_p_v, logvar_p_v = model(y_val, x_val)
                val_loss += cvae_loss(y_hat_v, y_val, mu_q_v, logvar_q_v,
                                      mu_p_v, logvar_p_v).item()

        avg_train = train_loss / len(train_loader.dataset)
        avg_val   = val_loss   / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: Train = {avg_train:.4f} | Val = {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best model saved to {MODEL_OUT}")


# Initialize model and set decoder bias for correct initial scale
model_sohn = SohnCVAE_Wendland(x_dim=3, W_alpha_basis=W_alpha_basis)
with torch.no_grad():
    nn.init.constant_(model_sohn.dec_fc2.bias, 3.0)

train_model(model_sohn, train_loader, val_loader)

# =============================================================================
# Emulation Function
# =============================================================================

def SohnCVAE_emulate(model, n_samples, condition_data):
    """
    Generate emulated spatial fields from the learned prior p(z | c).

    Args:
        model:          trained SohnCVAE_Wendland model
        n_samples:      int — number of emulated replicates
        condition_data: Tensor (n_t x 3) — ENSO condition inputs

    Returns:
        Tensor (n_loc x n_t x n_samples)
    """
    model.eval()
    device     = next(model.parameters()).device
    cond_input = condition_data.to(device)
    n_t_emu    = cond_input.shape[0]

    emulated = torch.zeros(2500, n_t_emu, n_samples)

    with torch.no_grad():
        h_p    = model.prior_net(cond_input)
        mu_p   = model.mu_p(h_p)
        std_p  = torch.exp(0.5 * model.logvar_p(h_p))

        for i in range(n_samples):
            z     = mu_p + torch.randn_like(std_p) * std_p
            y_hat = model.decode(z, cond_input)             # (n_t, 3, 2500)
            emulated[:, :, i] = y_hat[:, 1, :].cpu().T     # current time channel

            if (i + 1) % 500 == 0:
                print(f"Emulated {i + 1}/{n_samples}")

    return emulated


# Generate emulations
SohnCVAE_emulated        = SohnCVAE_emulate(model_sohn, 2000, x_cond)
SohnCVAE_emulated_counterfact = SohnCVAE_emulate(model_sohn, 2000, MEI_SCALE - x_cond)

white_noise = torch.rand_like(x_cond) * MEI_SCALE
SohnCVAE_emulated_whitenoise = SohnCVAE_emulate(model_sohn, 2000, white_noise)

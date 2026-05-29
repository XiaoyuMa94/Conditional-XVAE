# =============================================================================
# pretrain_CNN_FWI.py
#
# Pretrain a CNN to estimate the tilting parameter field theta_t from
# the latent expPS variables Z_t and the ENSO condition c_t for the
# FWI real data analysis in Ma et al. (2025).
#
# This pretrained CNN is used to warm-start the decoder of the cXVAE,
# giving it a better initial estimate of the dependence parameters.
# Running this script is strongly recommended before training the cXVAE.
#
# Prerequisites:
#   - data_preparation.R has been run and CSV outputs exist in DATA_DIR
#
# Output:
#   CNN_pretrained_FWI.pt  — saved CNN weights
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR  = "data/FWI"
MODEL_OUT = "CNN_pretrained_FWI.pt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ALPHA         = 0.5
MEI_SCALE     = 2.0
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 40
BATCH_SIZE    = 1      # stableloss processes one time point at a time

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

X_tensor  = torch.from_numpy(X).to(torch.float32)
W_tensor  = torch.tensor(W, dtype=torch.float32)
W_alpha   = W_tensor.pow(1 / ALPHA)
rbf_mat   = torch.tensor(RBF).to(torch.float32)

k          = W_alpha.shape[1]    # 540
knot_width = 20                  # 20x27 knot grid (k=540, not a perfect square)
alpha_size = rbf_mat.shape[1]
n_t        = X_tensor.shape[1]  # 127

# =============================================================================
# Z Approximation via Truncated SVD
# =============================================================================

# Project X onto the latent space using truncated pseudo-inverse of W_alpha
U, D, Vt = np.linalg.svd(W_alpha.numpy(), full_matrices=False)
keep      = D > (1e-2 * np.max(D))
U_k, D_k, V_k = U[:, keep], D[keep], Vt[keep, :].T
Z_approx  = torch.tensor(V_k @ np.diag(1 / D_k) @ U_k.T @ X).to(torch.float32)
Z_approx  = F.softplus(Z_approx, beta=8) + 1e-4

Z_tensor = Z_approx   # (k x n_t)

# =============================================================================
# Build CNN Inputs
# =============================================================================

rbf_vec = rbf_mat.reshape(1, -1)

Z_3d_ENSO, Z_3d = utils.aug_2k(
    dat=Z_tensor, meis=Nina34.T, rbf_vec=rbf_vec, width=knot_width
)

train_data   = utils.CustomDataset(Z_3d_ENSO, Z_3d)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# CNN Model
# =============================================================================

class CNN(nn.Module):
    """
    CNN that maps fused (log Z, MEI) spatial inputs to basis coefficients
    for the tilting parameter field theta_t.

    Input:  (batch x 3 x knot_height x 2*knot_width)  i.e. (B x 3 x 27 x 40)
    Output: (batch x alpha_size)
    """
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

        # Infer flattened size automatically
        with torch.no_grad():
            dummy    = torch.zeros(1, *Z_3d_ENSO.shape[1:])
            flat_dim = self._forward_conv(dummy).view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, alpha_size)

    def _forward_conv(self, x):
        x = self.pool(self.conv2(self.pool(self.conv1(x))))
        x = self.pool(self.conv4(self.pool(self.conv3(x))))
        x = self.pool(self.conv6(self.pool(self.conv5(x))))
        x = self.pool(self.conv8(self.pool(self.conv7(x))))
        return x

    def forward(self, x):
        x = torch.flatten(self._forward_conv(x), start_dim=1)
        return self.fc1(x).view(-1, alpha_size)


# =============================================================================
# Loss Function
# =============================================================================

def stableloss(outputs, data):
    """
    expPS log-likelihood loss for tilting parameter estimation.
    Minimizes: E[Z * theta - sqrt(theta)] over the latent expPS variables,
    with a temporal smoothness penalty on the basis coefficients.
    """
    alpha_coef = outputs.float()
    d1 = torch.diff(alpha_coef, dim=0)

    dt  = data[0, :, range(0, k)]
    rb  = data[0, 0, range(k, data.shape[2] - 1)].reshape(k, alpha_size).T
    ts  = data[0, :, data.shape[2] - 1]

    ts_1 = max((ts[0] - ts[1]).abs().item(), 3e-4)
    ts_2 = max((ts[1] - ts[2]).abs().item(), 3e-4)

    theta   = torch.relu(alpha_coef @ rb)
    loss_p1 = (-(theta.sqrt() - dt.exp() * theta)).sum().div(theta.numel())
    loss_p2 = (d1[0].div(ts_1).pow(2).sum().sqrt()
               + d1[1].div(ts_2).pow(2).sum().sqrt())
    loss_p2 = loss_p2.mul(0.001).div(d1.numel())

    return loss_p1, loss_p2


# =============================================================================
# Training
# =============================================================================

torch.manual_seed(12)
model     = CNN(alpha_size=alpha_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-8
)

loss_list = []

print(f"Training CNN on {DEVICE} for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    t = tqdm(enumerate(train_loader), total=len(train_loader),
             desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for i, (inputs, truth) in t:
        inputs, truth = inputs.to(DEVICE), truth.to(DEVICE)
        optimizer.zero_grad()
        outputs     = model(inputs)
        loss_p1, loss_p2 = stableloss(outputs, truth)
        loss        = loss_p1 + loss_p2
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        t.set_description(
            f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    avg_loss = epoch_loss / len(train_loader.dataset)
    scheduler.step(avg_loss)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.6f}")

# =============================================================================
# Save Model
# =============================================================================

torch.save(model.state_dict(), MODEL_OUT)
print(f"Pretrained CNN saved to {MODEL_OUT}")

# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(loss_list) + 1), loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Pretraining Loss (FWI)")
plt.tight_layout()
plt.savefig("pretrain_loss_FWI.png", dpi=150)
plt.close()
print("Loss curve saved to pretrain_loss_FWI.png")

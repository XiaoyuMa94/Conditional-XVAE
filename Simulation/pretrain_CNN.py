# =============================================================================
# pretrain_CNN.py
#
# Pretrain a CNN to estimate the tilting parameter field theta_t from
# the latent expPS variables Z_t and the ENSO condition c_t.
#
# This pretrained CNN is used to warm-start the decoder of the cXVAE,
# giving it a better initial estimate of the dependence parameters.
# Running this script is strongly recommended before training the cXVAE.
#
# Output:
#   CNN_pretrained.pt  — saved CNN weights
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

# =============================================================================
# Configuration — modify paths as needed
# =============================================================================

DATA_DIR  = "data/simulation"   # directory containing CSV outputs from simulate_data.R
MODEL_OUT = "CNN_pretrained.pt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 5e-7
NUM_EPOCHS    = 20
BATCH_SIZE    = 1       # stableloss processes one time point at a time
MEI_SCALE     = 2.0     # amplify ENSO signal
ALPHA         = 0.5

# =============================================================================
# Load Data
# =============================================================================

def load_csv(filename):
    return pd.read_csv(
        os.path.join(DATA_DIR, filename),
        header=None, skiprows=1
    ).drop([0], axis="columns").values


W       = load_csv("W_Data.csv")
Z       = load_csv("Z_Data.csv")
Thetas  = load_csv("Thetas_Data.csv")
RBF     = load_csv("RBF_Data.csv")
MEIs_MA = load_csv("MEIs_MA_Data.csv")

k          = W.shape[1]
alpha_size = RBF.shape[1]   # 144
knot_width = int(k ** 0.5)  # 15

MEIs_MA   = MEIs_MA * MEI_SCALE
n_t       = MEIs_MA.shape[0]

Z_tensor      = torch.tensor(Z)
RBF_tensor    = torch.tensor(RBF)
MEI_tensor    = torch.from_numpy(MEIs_MA)
Thetas_tensor = torch.from_numpy(Thetas)

rbf_vec = RBF_tensor.reshape(1, -1)

# Build CNN input and loss evaluation tensors
Z_3d_ENSO, Z_3d = utils.aug_2k(
    dat=Z_tensor, meis=MEI_tensor, rbf_vec=rbf_vec, width=knot_width
)

train_data   = utils.CustomDataset(Z_3d_ENSO, Z_3d)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# CNN Model
# =============================================================================

class CNN(nn.Module):
    """
    Convolutional network that maps fused (log Z, MEI) spatial inputs
    to basis coefficients for the tilting parameter field theta_t.

    Input:  (batch x 3 x K/width x 2*width)
    Output: (batch x alpha_size)
    """
    def __init__(self, alpha_size, input_shape=(3, 15, 30)):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32,  3, padding="same")
        self.conv2 = nn.Conv2d(32, 64,  3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv6 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv7 = nn.Conv2d(64, 32,  3, padding="same")
        self.conv8 = nn.Conv2d(32, 16,  3, padding="same")
        self.pool  = nn.MaxPool2d(2, 1)

        with torch.no_grad():
            dummy    = torch.zeros(1, *input_shape)
            flat_dim = self._forward_conv(dummy).view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, alpha_size)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.view(-1, alpha_size)


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
    d1  = torch.diff(alpha_coef, dim=0)

    dt  = data[0, 1, range(0, k)]
    rb  = data[0, 0, range(k, data.shape[2] - 1)].reshape(k, alpha_size).T
    ts  = data[0, :, data.shape[2] - 1]

    ts_1 = max((ts[0] - ts[1]).abs().item(), 0.003)
    ts_2 = max((ts[1] - ts[2]).abs().item(), 0.003)

    theta  = torch.relu(alpha_coef @ rb)
    loss_p1 = (dt.exp() * theta - theta.sqrt()).sum().div(dt.numel())
    loss_p2 = 0  # temporal penalty disabled; enable if needed

    return loss_p1 + loss_p2


# =============================================================================
# Training
# =============================================================================

torch.manual_seed(12)
model     = CNN(alpha_size=alpha_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-8
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
        outputs = model(inputs)
        loss    = stableloss(outputs, truth)
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

# =============================================================================
# Quick Diagnostic: theta prediction vs truth
# =============================================================================

model.eval()
test_alpha = torch.zeros(n_t, alpha_size)

with torch.no_grad():
    for t_idx in range(n_t):
        inp = Z_3d_ENSO[t_idx].unsqueeze(0).to(DEVICE)
        test_alpha[t_idx] = model(inp)[0].cpu()

theta_pred = test_alpha.double() @ RBF_tensor.T
theta_pred = theta_pred.relu()

mae = torch.mean((Thetas_tensor.T - theta_pred).abs())
print(f"Mean absolute error (theta): {mae:.6f}")

# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(loss_list) + 1), loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Pretraining Loss")
plt.tight_layout()
plt.savefig("pretrain_loss.png", dpi=150)
plt.close()
print("Loss curve saved to pretrain_loss.png")

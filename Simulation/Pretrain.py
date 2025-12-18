
import os
import torch.nn.functional as F
import utils
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


# Read data files
X = pd.read_csv('X_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
W = pd.read_csv('W_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
Thetas = pd.read_csv('Thetas_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
RBF = pd.read_csv('RBF_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
MEIs_MA = pd.read_csv('MEIs_MA_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
MEI_scale = 2
MEIs_MA = MEIs_MA * MEI_scale  # MEIs should be amplified to guarantee its influence

k = RBF.shape[0]
alpha_size = RBF.shape[1]
knot_width = int(k**0.5)
n_t = MEIs_MA.shape[0]
alpha = 0.5

# Convert the data from numpy arrays to tensors
X_tensor = torch.from_numpy(X)
W_tensor = torch.from_numpy(W)
W_alpha = W_tensor.pow(1 / alpha)
proj = W_alpha @ torch.linalg.inv(W_alpha.T @ W_alpha)
Z_approx = (X_tensor.T @ proj).T
Z_approx = utils.softplus_clip(Z_approx, beta=8)+0.0001

Z_tensor = Z_approx
Thetas_tensor = torch.from_numpy(Thetas)
RBF_tensor = torch.from_numpy(RBF)
MEI_tensor = torch.from_numpy(MEIs_MA)

# Vectorize the RBF
rbf_vec = RBF_tensor.reshape(1, -1)

# Generate the CNN inputs (data for train, data for loss evaluation) with the chessboard_aug() function
Z_3d_ENSO, Z_3d = utils.aug_2k(dat=Z_tensor, meis=MEI_tensor, rbf_vec=rbf_vec, width=knot_width)

n_train = 528
n_valid = 0
n_test = 528 - n_train - n_valid

# Split to train, validation, test
Z_train_3d_ENSO = Z_3d_ENSO[range(0, n_train), :]

Z_train_3d = Z_3d[range(0, n_train), :]

train_data = utils.CustomDataset(Z_train_3d_ENSO, Z_train_3d)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(60, 80, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(80, 100, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(3300, alpha_size*3)


    def forward(self, x):

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))


        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = x.view(-1, alpha_size)
        return x


# Define the customized loss function
def stableloss(outputs, data):
    alpha = outputs
    alpha = alpha.float()
    d1 = torch.diff(alpha, dim=0)
    dt = data[0, :, range(0, k)]
    rb = data[0, 0, range(k, data.shape[2]-1)].reshape(k, alpha_size).T
    ts = data[0, :, data.shape[2]-1]
    if ts[0] - ts[1] == 0:
        ts_1 = 0.003
    else:
        ts_1 = (ts[0] - ts[1]).abs()
    if ts[1] - ts[2] == 0:
        ts_2 = 0.003
    else:
        ts_2 = (ts[1] - ts[2]).abs()

    theta = torch.mm(alpha, rb).relu()
    weights = torch.tensor([1, 1, 1], device=dt.device).reshape(3, 1)
    theta_repeated = theta.expand(3, -1)

    # Now compute per-time likelihood terms
    per_time_loss = -(theta_repeated.sqrt() - dt.exp() * theta_repeated)
    loss_p1 = (per_time_loss * weights).mean()

    loss_p2 = d1[0,].div(ts_1).pow(2).sum().sqrt() + d1[1,].div(ts_2).pow(2).sum().sqrt()
    loss_p2 = loss_p2.mul(0.0001).div(d1.numel())

    loss = (loss_p1 + loss_p2)
    return loss

train_loader = DataLoader(train_data, batch_size=1, shuffle=False)

# Initialize model, loss function, and optimizer
torch.manual_seed(12)
model = CNN()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

# Training loop
loss_list = []
num_epochs = 10
torch.manual_seed(12)
for epoch in range(num_epochs):
    loss_1 = 0
    t = tqdm(
        enumerate(train_loader, 0), total=train_loader.__len__())
    for i, data in t:
        inputs, truth = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = stableloss(outputs, truth)

        loss.backward()
        optimizer.step()
        loss_1 += loss.item()

        t.set_description("Epoch: {}, Loss: {}, LR: {}".format(epoch+1, loss.item(), scheduler.get_last_lr()[0])
                          )
    scheduler.step()
    loss_list.append(loss_1)
loss_list = np.array(loss_list) / 528
print(loss_list)

# Specify a path
PATH = "CNN_10epochs.pt"

# Save
torch.save(model.state_dict(), PATH)


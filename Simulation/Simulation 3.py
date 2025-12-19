
import os
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio

matplotlib.use('TkAgg')

os.chdir("~/CXVAE")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
alpha = 0.5

# Read data files
X = pd.read_csv('X_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
Z = pd.read_csv('Z_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
Thetas = pd.read_csv('Thetas_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
RBF = pd.read_csv('RBF_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
MEIs_MA = pd.read_csv('MEIs_MA_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
y = pd.read_csv('Y_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values
W = pd.read_csv('W_Data.csv', header=None, skiprows=1).drop([0], axis='columns').values

k = W.shape[1]
knot_width = int(k ** 0.5)
latent_size = k * 3
hidden_size = k * 3
num_epochs = 500

learning_rate = 1e-6

n_loc, n_time = X.shape
n_holdout = 10


MEI_scale = 2
MEIs_MA = MEIs_MA * MEI_scale  # MEIs should be amplified to guarantee its influence
alpha_size = RBF.shape[1]
n_t = MEIs_MA.shape[0]
W_alpha = torch.tensor(W).pow(1 / alpha).to(torch.float32)
X_tensor = torch.tensor(X).to(torch.float32)
MEI_tensor = torch.tensor(MEIs_MA).T
rbf_mat = torch.tensor(RBF).to(torch.float32)
Y_tensor = torch.tensor(y).to(torch.float32)

# Select holdout locations with extremes

# X: shape [n_loc, n_time]
q = 0.9

# 1️⃣ Compute the global 90th percentile threshold
threshold = torch.quantile(X_tensor, q)

# 2️⃣ Count how many time steps per location exceed it
count_high = (X_tensor > threshold).sum(dim=1)   # shape [n_loc]

_, sorted_idx = torch.sort(count_high)
holdout_idx = sorted_idx[2480:2490]
train_idx = np.setdiff1d(np.arange(n_loc), holdout_idx)

# Stations
station_x = np.linspace(0, 20, 50)
station_y = np.linspace(0, 20, 50)
xx, yy = np.meshgrid(station_x, station_y)
stations = np.column_stack([xx.ravel(), yy.ravel()])

stations_train = stations[train_idx, :]
stations_holdout = stations[holdout_idx, :]

W_alpha = W_alpha.to(device)
X_tensor = X_tensor.to(device)
MEI_tensor = MEI_tensor.to(device)
rbf_mat = rbf_mat.to(device)
Y_tensor = Y_tensor.to(device)

X_train = X_tensor[train_idx, :]
X_holdout = X_tensor[holdout_idx, :]
W_train = W_alpha[train_idx, :]
W_holdout = W_alpha[holdout_idx, :]

X_train_tensor = utils.x_aug(X_train)
X_holdout_tensor = utils.x_aug(X_holdout)
MEI_input = utils.x_aug(MEI_tensor)

batch_size = X.shape[1]

train_data = utils.CVAEinput_Dataset(X_train_tensor, MEI_input)
holdout_data = utils.CVAEinput_Dataset(X_holdout_tensor, MEI_input)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
holdout_loader = DataLoader(dataset=holdout_data, batch_size=batch_size, shuffle=False)

image_size = X_train_tensor.shape[1]

proj = W_alpha @ torch.linalg.inv(W_alpha.T @ W_alpha)  # n.s x k
proj_holdout = proj[holdout_idx, :]

# P 0 0
# 0 P 0
# 0 0 P

proj_initial = proj[train_idx, :].T
proj_r1 = torch.cat([proj_initial,
                     torch.zeros(proj_initial.shape[0], proj_initial.shape[1]),
                     torch.zeros(proj_initial.shape[0], proj_initial.shape[1])])
proj_r2 = torch.cat([torch.zeros(proj_initial.shape[0], proj_initial.shape[1]),
                     proj_initial,
                     torch.zeros(proj_initial.shape[0], proj_initial.shape[1])])
proj_r3 = torch.cat([torch.zeros(proj_initial.shape[0], proj_initial.shape[1]),
                     torch.zeros(proj_initial.shape[0], proj_initial.shape[1]),
                     proj_initial])
proj_final = torch.cat([proj_r1, proj_r2, proj_r3], dim=1)
# torch.save(proj_final, 'proj_final.pt')
# proj_final = torch.load("proj_final.pt")

W_alpha_r1 = torch.cat([W_train.T,
                        torch.zeros(W_train.T.shape[0], W_train.T.shape[1]),
                        torch.zeros(W_train.T.shape[0], W_train.T.shape[1])])
W_alpha_r2 = torch.cat([torch.zeros(W_train.T.shape[0], W_train.T.shape[1]),
                        W_train.T,
                        torch.zeros(W_train.T.shape[0], W_train.T.shape[1])])
W_alpha_r3 = torch.cat([torch.zeros(W_train.T.shape[0], W_train.T.shape[1]),
                        torch.zeros(W_train.T.shape[0], W_train.T.shape[1]),
                        W_train.T])
W_alpha_final = torch.cat([W_alpha_r1, W_alpha_r2, W_alpha_r3], dim=1)
# torch.save(W_alpha_final, 'W_alpha_final.pt')
# W_alpha_final = torch.load("W_alpha_final.pt")

proj = proj.to(device)
proj_final = proj_final.to(device)
W_alpha_final = W_alpha_final.to(device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(60, 80, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(80, 100, kernel_size=3, padding="same")

        self.pool = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(3300, alpha_size * 3)

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


# Conditional VAE model

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
        self.softplus = nn.Softplus(beta=8)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(60, 80, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(80, 100, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(2, 1)
        self.fct = nn.Linear(3300, alpha_size * 3)
        self.reg_lin = nn.Linear(3, k * 3, bias=False)
        self.l1 = nn.Linear(latent_size * 2, image_size)

    def encode(self, x):
        h1 = self.softplus(self.fc1(x))
        mu = self.softplus(self.fc2(h1))
        log_var = self.fc3(h1)
        var = torch.exp(log_var)
        return mu, var

    def reparameterize(self, mu, var, label):
        eps = torch.normal(0, 1, size=(var.shape[0], var.shape[1]), device=var.device)
        mu = mu.log()
        reg_term = self.reg_lin(label)
        z = mu + torch.sqrt(var) * eps + reg_term
        return z, eps

    def decode(self, z, label):
        # CNN part
        z_inputs = utils.z_chess_aug_2k(z, label, width=knot_width).to(self.conv1.weight.device)
        t1 = self.pool(self.conv2(self.pool(self.conv1(z_inputs))))
        t1 = self.pool(self.conv4(self.pool(self.conv3(t1))))
        t2 = self.pool(self.conv5(t1))
        t3 = (self.fct(torch.flatten(t2, start_dim=1)))
        random_coefficients = t3.view(-1, alpha_size)

        z_origin = z_inputs.exp().reshape(z_inputs.shape[0], -1)
        y_star = self.softplus(self.l1(z_origin))

        return y_star, random_coefficients

    def forward(self, x, label):
        mu, var = self.encode(x)
        z, eps = self.reparameterize(mu, var, label)
        y_star, random_coefficients = self.decode(z, label)
        return y_star, random_coefficients, eps, var, z


torch.manual_seed(12)
model = CVAE().to(device)

# Initialize the weights
mask = torch.zeros_like(model.l1.weight)
mask[:, 0::2] = W_alpha_final.T
model.l1.weight = nn.parameter.Parameter(mask)
model.l1.bias = nn.parameter.Parameter(torch.zeros(model.l1.bias.shape[0]))

model.fc1.weight = nn.parameter.Parameter(proj_final)
model.fc1.bias = nn.parameter.Parameter(torch.zeros(model.fc1.bias.shape[0]) + 0.0001)
model.fc2.weight = nn.parameter.Parameter(torch.diag(torch.ones(model.fc2.weight.shape[0])))
model.fc2.bias = nn.parameter.Parameter(torch.zeros(model.fc2.bias.shape[0]))
model.fc3.weight = nn.parameter.Parameter(torch.zeros(model.fc3.weight.shape[0], model.fc3.weight.shape[1]))
model.fc3.bias = nn.parameter.Parameter(torch.zeros(model.fc3.bias.shape[0]) - 10)
model.reg_lin.weight = nn.parameter.Parameter(torch.zeros(model.reg_lin.weight.shape) + 0.01)

PATH = "CNN_10epochs.pt"
pretrained_CNN = CNN().to(device)
pretrained_CNN.load_state_dict(torch.load(PATH))

model.conv1 = pretrained_CNN.conv1
model.conv2 = pretrained_CNN.conv2
model.conv3 = pretrained_CNN.conv3
model.conv4 = pretrained_CNN.conv4
model.conv5 = pretrained_CNN.conv5

model.fct = pretrained_CNN.fc1


# Loss function
def loss_function(x, label, var, eps, y_star, rbf_mat, random_coefficients, z):
    # Part 1
    x_tmp = x
    standardized = x_tmp.div(y_star + 0.001)
    loss_p1 = standardized.log().abs().mul(30).sum().mul(-1).div(standardized.numel())

    # Part 2
    theta = torch.mm(rbf_mat, random_coefficients.T).relu()
    z = z.reshape(-1, k).T
    log_lik_z = theta.sqrt() - z.mul(1.5) - z.exp().mul(theta) - z.exp().mul(4).pow(-1)
    loss_p2 = torch.sum(log_lik_z).div(log_lik_z.numel())

    # Part 3
    loss_p3 = torch.sum(var.log()) + torch.sum(eps.pow(2)) * 0.5
    loss_p3 = loss_p3.div(eps.numel())

    # Part 4
    r_tmp = random_coefficients.reshape(-1, 3, alpha_size)
    d1 = torch.diff(r_tmp, dim=1)
    ts = torch.diff(label, dim=1).abs()
    ts[ts == 0] = 0.0003

    loss_p4 = (0.05) * (d1.div(ts.repeat_interleave(alpha_size).reshape(-1, 2, alpha_size)).pow(2).sum().sqrt())
    loss_p4 = loss_p4.div(d1.numel())

    return loss_p1, loss_p2, loss_p3, loss_p4


# Optimizer
optimizer = optim.Adam([
    {"params": [p for n, p in model.named_parameters() if n != "l1.weight"],
     "lr": learning_rate},  # group 0: rest of model
    {"params": [model.l1.weight], "lr": 1e-4},  # group 1: l1.weight
])
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.2)

loss_list = []
loss_list_1 = []
loss_list_2 = []
loss_list_3 = []
loss_list_4 = []

# Record the start time
start_time = time.time()

for epoch in range(num_epochs):
    train_loss = 0
    train_loss_1 = 0
    train_loss_2 = 0
    train_loss_3 = 0
    # train_loss_4 = 0
    t = tqdm(
        enumerate(train_loader, 0),
        total=train_loader.__len__(),
        desc=f'Epoch {epoch + 1}/{num_epochs}')
    for i, (data, label) in t:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y_star, random_coefficients, eps, var, z = model(data, label)
        loss_p1, loss_p2, loss_p3, loss_p4 = loss_function(
            data, label, var, eps, y_star, rbf_mat, random_coefficients, z
        )
        loss = (loss_p1 + loss_p2 + loss_p3).mul(-1) + loss_p4
        loss.backward()
        with torch.no_grad():
            model.l1.weight.grad[:, 0::2] = 0  # freeze the update on weight correspond to W, so that fix W
        train_loss += loss.item()
        train_loss_1 += loss_p1.item()
        train_loss_2 += loss_p2.item()
        train_loss_3 += loss_p3.item()
        train_loss_4 += loss_p4.item()
        optimizer.step()

        t.set_description(
            "Epoch: {}, Loss: {}, LR: {}".format(epoch + 1, loss.item(), scheduler.get_last_lr()[0])
        )
    # Print learning rates for debugging
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Epoch {epoch}, Group {i} lr = {param_group['lr']:.6f}")

    scheduler.step()
    loss_list.append(train_loss)
    loss_list_1.append(train_loss_1)
    loss_list_2.append(train_loss_2)
    loss_list_3.append(train_loss_3)
    loss_list_4.append(train_loss_4)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

PATH = "CVAE_500epochs.pt" # holdout

# Save
torch.save(model.state_dict(), PATH)

# Load
model = CVAE()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

model.eval()
with torch.no_grad():
    y_star, random_coefficients, eps, var, z = model(train_loader.dataset.x, train_loader.dataset.y)

y_star = y_star[:, 2490:4980]

ind = np.round(np.linspace(1, 1582, 528)).astype(int)
test_alpha = random_coefficients[ind, :]

theta_pred = test_alpha @ torch.transpose(rbf_mat, 0, 1)
theta_pred = theta_pred.relu()
theta_test_pred = theta_pred[420:528, :]
theta_test = torch.tensor(Thetas[:, 420:528])
X_test_true = torch.tensor(X[:, 420:528])

err_1 = torch.empty_like(y_star).exponential_(30)
err_2 = torch.empty_like(y_star).bernoulli_(0.5) * 2 - 1

X_emu = err_1.mul(err_2).exp().mul(y_star).to(torch.float32)

##### Plotting #####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

fig, axs = plt.subplots(2, 5, figsize=(12, 6))

# Time labels for columns
time_labels = [f"Time {i + ind + 1}" for i in range(5)]

# Define bins
vmin, vmax = -5, 0
levels = np.linspace(vmin, vmax, 9)  # internal bins

# Replace infinities with large finite numbers for plotting
big = 1e6
boundaries = np.concatenate(([-big], levels, [big]))

# Discrete colormap
cmap = plt.get_cmap("Spectral", len(boundaries) - 1)
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

# --- Plotting ---
for i in range(5):
    img = images_sub_1[i].reshape(-1, 50).T.detach().numpy()
    im = axs[0, i].imshow(np.flip(img, 0), cmap=cmap, norm=norm)
    axs[0, i].set_title(time_labels[i])
    axs[0, i].axis("off")

for i in range(5):
    img = images_sub_2[i].reshape(-1, 50).T.detach().numpy()
    im = axs[1, i].imshow(np.flip(img, 0), cmap=cmap, norm=norm)
    axs[1, i].set_title(time_labels[i])
    axs[1, i].axis("off")

# Labels
fig.text(0.5, 0.9, "Truth", ha="center", va="center", fontsize=12, fontweight="bold")
fig.text(0.5, 0.52, "CVAE Emulations", ha="center", va="center", fontsize=12, fontweight="bold")

# Layout
plt.subplots_adjust(top=0.85, bottom=0.15, right=0.85)

# --- Colorbar ---
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, boundaries=boundaries, spacing="uniform")

# Tick positions: bin centers
tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
cbar.set_ticks(tick_positions)

# Interval labels (replace extremes with ∞)
tick_labels = [f"(-∞,{levels[0]:.1f}]"]
tick_labels += [f"({levels[i - 1]:.1f},{levels[i]:.1f}]" for i in range(1, len(levels))]
tick_labels.append(f"({levels[-1]:.1f},∞)")
cbar.set_ticklabels(tick_labels)

cbar.set_label("log(X)", rotation=270, labelpad=15)
fig.savefig("logx.png")
plt.show()

# Define the longitude and latitude ranges and step sizes
lon_start = 0
lon_end = 20
lon_step = 20 / 49

lat_start = 0
lat_end = 20
lat_step = 20 / 49

# Create the longitude and latitude vectors
lon_values = torch.arange(lon_start, lon_end + lon_step, lon_step)
lat_values = torch.arange(lat_start, lat_end + lat_step, lat_step)

# Create a grid of longitude and latitude values
lon_grid, lat_grid = torch.meshgrid(lon_values, lat_values, indexing='xy')

# Convert to numpy for plotting
lon = lon_grid.numpy()
lat = lat_grid.numpy()

# Flatten the grids and stack them to create a (2500, 2) tensor
location_tensor = torch.stack((lon_grid.flatten(), lat_grid.flatten()), dim=-1)
X_train_true = torch.tensor(X[:, :])


def chi_est(Data, Loc, d, tol=1e-2, gridded=False,
            plot_pairs=True, plot_chi=True,
            u_vec=torch.cat([torch.arange(0.95, 0.98, 0.01),
                             torch.arange(0.9801, 0.9997, 0.0001)]),
            CDF_fun=None):
    """
    Chi estimation for spatial extremes

    Args:
        Data: Tensor of shape (n_locations, n_timesteps)
        Loc: Tensor of shape (n_locations, 2) with coordinates
        d: Base distance threshold
        tol: Distance tolerance
        gridded: Whether locations are gridded
        grid_unit_length: Grid spacing unit
        plot_pairs: Whether to plot location pairs
        plot_chi: Whether to plot chi(u)
        u_vec: Quantile levels to evaluate
        CDF_fun: Custom CDF function (optional)

    Returns:
        Dictionary with chi estimates and confidence intervals
    """
    # Compute pairwise distances
    Dist = torch.cdist(Loc, Loc, p=2)

    # Find pairs within distance tolerance
    mask = (Dist > (d - tol)) & (Dist < (d + tol))
    pairs = torch.nonzero(mask)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # Remove duplicates

    # Filter pairs based on coordinate differences
    if (len(pairs) * Data.shape[1]) > 500:
        dx = Loc[pairs[:, 1], 0] - Loc[pairs[:, 0], 0]
        dy = Loc[pairs[:, 1], 1] - Loc[pairs[:, 0], 1]
        valid_pairs = (dx > 0) & (dy > 0)
        pairs = pairs[valid_pairs]

    # Additional gridded filtering
    if gridded and (len(pairs) > 500):
        pairs = pairs[::2]  # Take every other pair

    import random
    ind = random.sample(range(1, len(pairs)), 200)
    pairs = pairs[ind, :]
    # Plot location pairs
    if plot_pairs:
        plt.figure(figsize=(8, 6))
        plt.scatter(Loc[:, 0].cpu(), Loc[:, 1].cpu())
        for i, j in pairs:
            plt.plot([Loc[i, 0], Loc[j, 0]], [Loc[i, 1], Loc[j, 1]], 'r-')
        plt.title(f"Pairs with distance {d:.2f} ± {tol}")
        plt.show()

    # Extract dependent pairs
    n_times = Data.shape[1]
    expanded_pairs = pairs.repeat(n_times, 1)
    times = torch.arange(n_times).repeat_interleave(len(pairs))

    dep_pairs = torch.stack([
        Data[expanded_pairs[:, 0], times],
        Data[expanded_pairs[:, 1], times]
    ], dim=1)

    # Convert to uniform margins
    if CDF_fun:
        U_pairs = CDF_fun(dep_pairs)
    else:
        U_pairs = torch.zeros_like(dep_pairs)
        for col in [0, 1]:
            sorted_vals, _ = torch.sort(dep_pairs[:, col])
            ranks = torch.searchsorted(sorted_vals, dep_pairs[:, col], right=True)
            U_pairs[:, col] = ranks.float() / len(dep_pairs[:, col])

    # Calculate minimum values
    Min_sim = torch.min(U_pairs, dim=1)[0]

    # Compute chi estimates
    results = torch.zeros((len(u_vec), 3))

    for i, u in enumerate(u_vec):
        const = 0
        num_tmp = (Min_sim > u).float()  # A vec
        p_tmp_sim = num_tmp.sum().add(const).div(len(num_tmp) + const)  # A_bar
        denom_tmp = (U_pairs[:, 0] > u).float()  # B vec
        p_tmp1_sim = denom_tmp.sum().add(const).div(len(denom_tmp) + const)  # B_bar

        if p_tmp1_sim == 0 or p_tmp_sim == 0:
            results[i] = torch.tensor([0.0, 0.0, 0.0])
        else:
            ratio = num_tmp.mean() / denom_tmp.mean()

            var_A = p_tmp_sim * (1 - p_tmp_sim) / len(num_tmp)
            var_B = p_tmp1_sim * (1 - p_tmp1_sim) / len(denom_tmp)
            cov_AB = ((num_tmp - p_tmp_sim) * (denom_tmp - p_tmp1_sim)).mean() / len(num_tmp)
            var_r = var_A / p_tmp1_sim ** 2 + p_tmp_sim ** 2 * var_B / p_tmp1_sim ** 4 - 2 * p_tmp_sim * cov_AB / p_tmp1_sim ** 3

            std_sim = torch.sqrt(var_r.clamp(min=0))
            lower = (ratio - 1.96 * std_sim).clamp(min=0, max=1)
            upper = (ratio + 1.96 * std_sim).clamp(min=0, max=1)
            results[i] = torch.tensor([lower, upper, ratio])

    # Plot results
    if plot_chi:
        plt.figure(figsize=(8, 6))
        plt.plot(u_vec, results[:, 2], 'r-', label='Truth')
        plt.fill_between(u_vec.cpu(),
                         results[:, 0].cpu(),
                         results[:, 1].cpu(),
                         color='orange', alpha=0.3, label='95% CI')
        plt.ylim(0, 1)
        plt.xlabel('Quantile (u)')
        plt.ylabel('χ(u)')
        plt.legend()
        plt.title('Dependence Measure χ(u)')
        plt.show()

    return {
        'u': u_vec.cpu().numpy(),
        'truth': results[:, 2].cpu().numpy(),
        'upper': results[:, 1].cpu().numpy(),
        'lower': results[:, 0].cpu().numpy()
    }


u_vec = torch.cat([torch.arange(0.95, 0.98, 0.001),
                   torch.arange(0.981, 0.999, 0.0001)])

distance_vec = torch.tensor([0.5, 3, 6])
distances = torch.tensor([0.5, 3, 6])

comparisons = {}
com_ind = -1

for distance in distance_vec:
    com_ind = com_ind + 1
    chi_truth = chi_est(Data=X_train_true, Loc=location_tensor,
                        d=distance, tol=1e-1, gridded=True,
                        plot_pairs=False, plot_chi=False,
                        u_vec=u_vec,
                        CDF_fun=None)
    chi_tensor = torch.zeros(10, 3, len(u_vec))

    for iter in range(chi_tensor.shape[0]):
        print(iter)
        model.eval()
        with torch.no_grad():
            y_star, random_coefficients, eps, var, z = model(train_loader.dataset.x, train_loader.dataset.y)
        y_star_train = y_star[:, 1118:2236]
        err_1 = torch.empty_like(y_star_train).exponential_(30)
        err_2 = torch.empty_like(y_star_train).bernoulli_(0.5) * 2 - 1
        X_emu = err_1.mul(err_2).exp().mul(y_star_train).to(torch.float64)

        tmp = chi_est(Data=X_emu.T, Loc=location_tensor,
                      d=distance, tol=1e-1, gridded=True,
                      plot_pairs=False, plot_chi=False,
                      u_vec=u_vec,
                      CDF_fun=None)
        chi_tensor[iter, 0, :] = torch.tensor(tmp["truth"])
        chi_tensor[iter, 1, :] = torch.tensor(tmp["upper"])
        chi_tensor[iter, 2, :] = torch.tensor(tmp["lower"])

    chi_emu = torch.mean(chi_tensor[:, 0, :], dim=0)
    lower = torch.mean(chi_tensor[:, 2, :], dim=0)
    upper = torch.mean(chi_tensor[:, 1, :], dim=0)
    comparisons[com_ind] = chi_truth
    comparisons[com_ind]["emu"] = chi_emu
    comparisons[com_ind]["emu_lower"] = lower
    comparisons[com_ind]["emu_upper"] = upper

plt.figure(figsize=(8, 6))

linestyles = ["-", "--", ":"]

for i, ((distance, chi_truth)) in enumerate(sorted(comparisons.items())):
    if chi_truth is None:
        continue
    required_keys = ["u", "truth", "lower", "upper", "emu", "emu_lower", "emu_upper"]
    if not all(k in chi_truth for k in required_keys):
        continue

    ls = linestyles[i % len(linestyles)]

    # Plot curves
    plt.plot(chi_truth["u"], chi_truth["truth"], color='r', linestyle=ls, linewidth=1.5)
    plt.fill_between(chi_truth["u"], chi_truth["lower"], chi_truth["upper"], color='red', alpha=0.1)

    plt.plot(chi_truth["u"], chi_truth["emu"], color='b', linestyle=ls, linewidth=1.5)
    plt.fill_between(chi_truth["u"], chi_truth["emu_lower"], chi_truth["emu_upper"], color='blue', alpha=0.1)

plt.text(0.15, 0.82, "Short-range", fontsize=12, color='black', transform=plt.gca().transAxes)
plt.text(0.15, 0.46, "Medium-range", fontsize=12, color='black', transform=plt.gca().transAxes)
plt.text(0.15, 0.18, "Long-range", fontsize=12, color='black', transform=plt.gca().transAxes)
plt.ylim(0, 1)
plt.xlabel('Quantile (u)', fontsize=16)
plt.ylabel('χ(u)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.title("χ(u) Comparisons Across Distances", fontsize=16)
plt.tight_layout()
plt.savefig("chi_whole.png", dpi=300)
plt.show()


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

target_time_vec = torch.tensor([215, 221, 229])
theta_dat = torch.cat([torch.tensor(Thetas[:, target_time_vec - 1]), theta_pred[target_time_vec - 1, :].T], dim=1)
tmp_dat = torch.cat([X_tensor[:, target_time_vec - 1], X_emu[target_time_vec - 1, :].T], dim=1)

# --- Plot setup ---
fig, axes = plt.subplots(4, 3, figsize=(11, 13))
axes = axes.reshape(4, 3)

column_titles = ['December 1997', 'June 1998', 'February 1999']
row_labels = [r'$\theta$ (True)', r'$\theta$ (Emulated)',
              r'$\log(X)$ (True)', r'$\log(X)$ (Emulated)']

# --- Colormap for log(X) ---
vmin, vmax = -5, 0
levels = np.linspace(vmin, vmax, 9)
boundaries = np.concatenate(([-1e6], levels, [1e6]))
cmap = plt.get_cmap("Spectral", len(boundaries) - 1)
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

# --- Fill plots ---
for row in range(4):
    for col in range(3):
        ax = axes[row, col]

        if row < 2:  # θ data
            data = theta_dat[:, row * 3 + col].reshape(8, 8).T.detach().numpy()
            im = ax.imshow(np.flip(data, axis=0), cmap='plasma', vmin=0, vmax=2)
            if row == 0:  # save one reference for colorbar
                im_theta = im;
                ax.set_title(column_titles[col], fontsize=18)
        else:  # log(X) data
            data = tmp_dat[:, (row - 2) * 3 + col].log().reshape(50, 50).T.detach().numpy()
            im = ax.imshow(np.flip(data, axis=0), cmap=cmap, norm=norm)
            if row == 2:  # save one reference for colorbar
                im_logx = im

        ax.axis('off')

        # Column titles (top row only)
        if row == 0:
            ax.set_title(column_titles[col], fontsize=18)

# --- Row labels ---
row_pos = [0.88, 0.64, 0.39, 0.15]
for label, ypos in zip(row_labels, row_pos):
    fig.text(0.02, ypos, label, va='center', ha='left',
             fontsize=18, rotation=90)

# --- Colorbar for θ ---
theta_levels = np.linspace(0, 2, 11)
theta_ticks = (theta_levels[:-1] + theta_levels[1:]) / 2
theta_labels = [f"[0,{theta_levels[1]:.2f}]"] + \
               [f"({theta_levels[i]:.2f},{theta_levels[i + 1]:.2f}]"
                for i in range(1, len(theta_levels) - 1)]

cbar_ax_theta = fig.add_axes([0.86, 0.55, 0.012, 0.35])
cbar_theta = fig.colorbar(im_theta, cax=cbar_ax_theta,
                          boundaries=theta_levels, ticks=theta_ticks)
cbar_theta.set_ticklabels(theta_labels)
cbar_theta.ax.tick_params(labelsize=14)
cbar_theta.set_label(r'$\theta$', fontsize=16)

# --- Colorbar for log(X) ---
tick_positions = (boundaries[:-1] + boundaries[1:]) / 2
tick_labels = [f"(-∞,{levels[0]:.1f}]"] + \
              [f"({levels[i - 1]:.1f},{levels[i]:.1f}]"
               for i in range(1, len(levels))] + \
              [f"({levels[-1]:.1f},∞)"]

cbar_ax_x = fig.add_axes([0.86, 0.1, 0.012, 0.35])
cbar_x = fig.colorbar(im_logx, cax=cbar_ax_x,
                      boundaries=boundaries, ticks=tick_positions)
cbar_x.set_ticklabels(tick_labels)
cbar_x.ax.tick_params(labelsize=14)
cbar_x.set_label(r'$\log(X)$', fontsize=16)

# --- Layout ---
plt.subplots_adjust(left=0.06, right=0.83, top=0.95,
                    bottom=0.05, wspace=0.10, hspace=0.12)
plt.savefig('sim_theta_logX.png')
plt.show()

X_train_true = torch.tensor(X[:, :])
u_vec = torch.cat([torch.arange(0, 0.99, 0.01),
                   torch.arange(0.981, 0.999, 0.001)])
# u_vec = torch.cat([torch.arange(0, 1, 0.01)])
ARE_truth = torch.zeros([3, u_vec.shape[0]])
for iter in range(u_vec.shape[0]):
    ARE_truth[:, iter] = utils.ARE(u_vec[iter], unit_area=400 / 2401, x=X_train_true)

n_reps = 20
ns = 2500
nt = 528
ARE_res = torch.zeros((n_reps, 3, u_vec.shape[0]))

Emu_Data = torch.zeros([n_reps, ns, nt])
for iter in range(n_reps):
    print(iter)
    model.eval()
    with torch.no_grad():
        # y_star, random_coefficients, eps, var, z = model(test_loader.dataset.x, test_loader.dataset.y)
        y_star, random_coefficients, eps, var, z = model(train_loader.dataset.x, train_loader.dataset.y)
    y_star_train = y_star[:, 2500:5000]
    err_1 = torch.empty_like(y_star_train).exponential_(30)
    err_2 = torch.empty_like(y_star_train).bernoulli_(0.5) * 2 - 1
    X_emu = err_1.mul(err_2).exp().mul(y_star_train).to(torch.float64)
    Emu_Data[iter, :, :] = X_emu.T

for rep in range(n_reps):
    print(f"Processing replication {rep + 1}/{n_reps}")
    # ARE_res[rep] = torch.stack([ARE(u, unit_area, Data[rep, :, :]) for u in u_vec])
    ARE_emu = torch.zeros([3, u_vec.shape[0]])
    for iter in range(u_vec.shape[0]):
        ARE_emu[:, iter] = utils.ARE(u_vec[iter], unit_area=400 / 2401, x=Emu_Data[rep, :, :])
    ARE_res[rep, :, :] = ARE_emu
# Use mean as the final estimates
ARE_mean = ARE_res[:, 0, :].mean(dim=0)
low = ARE_res[:, 1, :].mean(dim=0)
upp = ARE_res[:, 2, :].mean(dim=0)

plt.figure(figsize=(8, 5))
plt.plot(u_vec, ARE_truth[0, :], 'r-', label="Truth")
plt.fill_between(u_vec, ARE_truth[1, :], ARE_truth[2, :], color='tomato', alpha=0.5, label="95% CI (Truth)")
plt.plot(u_vec, ARE_mean, 'b-', label="Emulation")
plt.fill_between(u_vec, low, upp, color='lightblue', alpha=0.5, label="95% CI (Emulation)")
# plt.title("ARE Plot", fontsize=18)
plt.xlabel("Quantile", fontsize=18)
plt.ylabel("Averaged Radius of Exceedances", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
# Adjust margins: reduce right whitespace
plt.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.12)
plt.legend(fontsize=16), plt.grid(True), plt.savefig("ARE plot.png"), plt.show()


from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import time

X_emu_sample = torch.zeros((500, 30, 2500))
X_emu_fake_sample = torch.zeros_like(X_emu_sample)
cand_ind = range(200, 230)
start_time = time.time()
for iter in range(X_emu_sample.shape[0]):
    print(iter)
    fake_y = MEI_scale - train_loader.dataset.y
    model.eval()
    with torch.no_grad():
        y_star, random_coefficients, eps, var, z = model(train_loader.dataset.x[cand_ind, :],
                                                         train_loader.dataset.y[cand_ind, :])
        fake_y_star, fake_random_coefficients, eps, var, fake_z = model(train_loader.dataset.x[cand_ind, :],
                                                                        fake_y[cand_ind, :])
    y_star_test = y_star[:, 2500:5000]
    err_1 = torch.empty_like(y_star_test).exponential_(30)
    err_2 = torch.empty_like(y_star_test).bernoulli_(0.5) * 2 - 1
    X_emu_sample[iter, :, :] = err_1.mul(err_2).exp().mul(y_star_test)

    fake_y_star_test = fake_y_star[:, 2500:5000]
    err_1 = torch.empty_like(fake_y_star_test).exponential_(30)
    err_2 = torch.empty_like(fake_y_star_test).bernoulli_(0.5) * 2 - 1
    X_emu_fake_sample[iter, :, :] = err_1.mul(err_2).exp().mul(fake_y_star_test)
end_time = time.time()  # Record end time
print(f"Code executed in {end_time - start_time:.4f} seconds")


def plot():
    # Create a list to store the images
    images = []

    # Loop through each frame and save the figure as an image
    for i in range(30):
        time_ind = i + cand_ind[1]
        sample1 = X_emu_fake_sample.detach().reshape(-1, 30, 50, 50)[:, i, 22, 8]  # loc 1, counter
        sample2 = X_emu_fake_sample.detach().reshape(-1, 30, 50, 50)[:, i, 18, 14]  # loc 2, counter
        sample3 = X_emu_sample.detach().reshape(-1, 30, 50, 50)[:, i, 22, 8]  # loc 1, CVAE
        sample4 = X_emu_sample.detach().reshape(-1, 30, 50, 50)[:, i, 18, 14]  # loc 2, CVAE

        # Create a grid for the contour plots
        x1 = np.linspace(min(sample1), max(sample1), 100)
        y1 = np.linspace(min(sample2), max(sample2), 100)
        X1, Y1 = np.meshgrid(x1, y1)

        x2 = np.linspace(min(sample3), max(sample3), 100)
        y2 = np.linspace(min(sample4), max(sample4), 100)
        X2, Y2 = np.meshgrid(x2, y2)

        # Compute the density for the first pair (sample1 and sample2)
        positions1 = np.vstack([X1.ravel(), Y1.ravel()])
        values1 = np.vstack([sample1, sample2])
        kernel1 = gaussian_kde(values1, bw_method=0.8)  # larger more smooth
        Z1 = np.reshape(kernel1(positions1).T, X1.shape)

        # Compute the density for the second pair (sample3 and sample4)
        positions2 = np.vstack([X2.ravel(), Y2.ravel()])
        values2 = np.vstack([sample3, sample4])
        kernel2 = gaussian_kde(values2, bw_method=0.8)
        Z2 = np.reshape(kernel2(positions2).T, X2.shape)
        # Normalize KDE to max=1 for relative density levels
        Z1_norm = Z1 / Z1.max()
        Z2_norm = Z2 / Z2.max()

        # Plot the contour plots on the same figure
        plt.figure(figsize=(7, 7))
        levels = np.linspace(0.1, 1, 10)
        # Contour plot for sample1 and sample2
        contour1 = plt.contour(X1, Y1, Z1_norm, levels=levels, colors='blue',
                               linewidths=1, linestyles='solid')
        plt.clabel(contour1, inline=True, fontsize=12, fmt='%.1f')  # label levels

        # Contour plot for sample3 and sample4
        contour2 = plt.contour(X2, Y2, Z2_norm, levels=levels, colors='red',
                               linewidths=1, linestyles='solid')
        plt.clabel(contour2, inline=True, fontsize=12, fmt='%.1f')
        # Create custom legend entries
        legend_elements = [
            Line2D([0], [0], color='red', lw=1, linestyle='solid', label='Counterfact'),
            Line2D([0], [0], color='blue', lw=1, linestyle='solid', label='CVAE Emulation')
        ]

        # Add labels, title, and legend
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.legend(handles=legend_elements, loc='upper left', fontsize=20)
        plt.show()

        # Save the plot as an image
        filename = f'Dist_Time_{time_ind}.png'
        plt.savefig(filename)
        plt.close()

        # Append the image to the list
        images.append(imageio.imread(filename))

    # Save the list of images as a GIF
    imageio.mimsave('Contour_ani.gif', images, fps=1)


if __name__ == "__main__":
    plot()


# # Generate sample data
x = np.arange(460, 490) + 1
y1 = MEIs_MA[461:491] / MEI_scale
y2 = 1 - y1

# Generate month labels from 2018-05 to 2020-10 (length must match x)
months = pd.date_range(start='2018-05-01', end='2020-10-01', freq='MS')
month_labels = months.strftime('%Y/%m')

# Create the plot
plt.figure(figsize=(10, 5))

plt.scatter(x, y1, color='blue', label='ENSO', marker='o', s=50)
plt.scatter(x, y2, color='red', label='Counterfacts', marker='o', s=50)

# Add vertical lines
vertical_lines = [468, 475, 488]
for line_x in vertical_lines:
    plt.axvline(x=line_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Replace x-axis ticks with month labels
plt.xticks(ticks=x[::3], labels=month_labels[::3], rotation=45)  # Show every 3rd month

plt.xlabel('Time', fontsize=12)
plt.ylabel('ENSO', fontsize=12)
plt.legend(loc="upper left", fontsize=14)

plt.tight_layout()
plt.savefig("MEI_scatter.png")
plt.show()


x = np.arange(200, 230) + 1
y1 = MEIs_MA[200:230] / MEI_scale
y2 = 1 - y1

# Generate month labels from 2018-05 to 2020-10 (length must match x)
months = pd.date_range(start='1996-10-01', end='1999-03-01', freq='MS')
month_labels = months.strftime('%B %Y')

# Create the plot
plt.figure(figsize=(10, 5))

plt.scatter(x, y1, color='blue', label='ENSO', marker='o', s=50)
plt.scatter(x, y2, color='red', label='Counterfacts', marker='o', s=50)

# Add vertical lines
vertical_lines = [215, 221, 229]
for line_x in vertical_lines:
    plt.axvline(x=line_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Replace x-axis ticks with month labels
ticks = [x[0]] + list(x[::3]) + [x[-1]]
labels = [month_labels[0]] + list(month_labels[::3]) + [month_labels[-1]]

plt.xticks(ticks=ticks, labels=labels, rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('ENSO', fontsize=12)
plt.legend(loc="upper left", fontsize=14)

plt.tight_layout()
plt.savefig("MEI_scatter200.png")
plt.show()

## Metrics
from scipy.interpolate import griddata

def empirical_cdf(x):
    x = x.flatten()
    sorted_x, _ = torch.sort(x)
    n = len(x)

    def cdf_fn(v):
        v = v.flatten()
        # Count number of elements <= each value in v
        # torch.searchsorted returns insertion indices in sorted_x
        idx = torch.searchsorted(sorted_x, v, right=True)
        return idx.float() / n

    return cdf_fn

def quantilefun(y):
    y_sorted, _ = torch.sort(y)
    n = len(y)
    p = torch.linspace(0, 1, n)

    def f(q):
        q = torch.clamp(q, 0.0, 1.0)  # ensure within [0, 1]
        idx = torch.bucketize(q, p)
        idx = torch.clamp(idx, 1, n - 1)
        x0, x1 = p[idx - 1], p[idx]
        y0, y1 = y_sorted[idx - 1], y_sorted[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (q - x0)

    return f

def qq_plot_quantile_scaled_custom(x, y, n_points=100, pch=20, quantile=True, ax=None, lim=10):

    # Ensure consistent dtype and device
    dtype = x.dtype
    device = x.device

    # Remove NaNs
    mask = ~torch.isnan(x) & ~torch.isnan(y)
    x = x[mask]
    y = y[mask]

    # Sort
    sort_x, _ = torch.sort(x)
    sort_y, _ = torch.sort(y)

    # Define equally spaced probabilities on the same dtype/device
    p = torch.linspace(0, 1, n_points, dtype=dtype, device=device)

    # Compute quantiles safely
    xq = torch.quantile(sort_x, p)
    yq = torch.quantile(sort_y, p)

    # ---- Confidence bands (Kolmogorov-style) ----
    m, n = len(x), len(y)
    N = m + n
    M = m * (n / N)
    K = 1.36
    qy = lambda q: torch.quantile(sort_y, torch.clamp(q, 0, 1))
    yl = qy(p - K / np.sqrt(M))
    yu = qy(p + K / np.sqrt(M))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot points and bands
    ax.scatter(xq.cpu(), yq.cpu(), s=pch, color="black")
    ax.plot(xq.cpu(), yl.cpu(), "--", color="gray", label="95% band")
    ax.plot(xq.cpu(), yu.cpu(), "--", color="gray")
    ax.plot(xq.cpu(), xq.cpu(), "--", color="darkorange", label="1-1 line")

    ax.set_xlabel("Truth")
    ax.set_ylabel("Emulation")
    ax.grid(True, linestyle=":")
    ax.legend()

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    plt.show()
    return {"p": p, "xq": xq, "yq": yq, "lower": yl, "upper": yu}

def spatial_prediction(model, n_samples=2000):
    values_1 = model.l1.weight[0:2490, :].detach().numpy()  # one value vector per location
    values_2 = model.l1.weight[2490:4980, :].detach().numpy()
    values_3 = model.l1.weight[4980:7470, :].detach().numpy()
    values_4 = model.l1.bias[0:2490].detach().numpy()
    values_5 = model.l1.bias[2490:4980].detach().numpy()
    values_6 = model.l1.bias[4980:7470].detach().numpy()

    # Interpolate (linear or nearest)
    interp_vals_1 = griddata(points, values_1, new_points, method='linear')
    interp_vals_2 = griddata(points, values_2, new_points, method='linear')
    interp_vals_3 = griddata(points, values_3, new_points, method='linear')
    interp_vals_4 = griddata(points, values_4, new_points, method='linear')
    interp_vals_5 = griddata(points, values_5, new_points, method='linear')
    interp_vals_6 = griddata(points, values_6, new_points, method='linear')

    interp_weight = np.concatenate((interp_vals_1, interp_vals_2, interp_vals_3), axis=0)
    interp_bias = np.concatenate((interp_vals_4, interp_vals_5, interp_vals_6), axis=0)
    interp_weight = torch.tensor(interp_weight).to(torch.float32)
    interp_bias = torch.tensor(interp_bias).to(torch.float32)

    sample_1 = torch.zeros([10, 528, n_samples])
    sample_2 = torch.zeros([10, 528, n_samples])

    for i in range(n_samples):
        model.eval()
        with torch.no_grad():
            # Encode inputs
            mu, var = model.encode(train_loader.dataset.x)

            # Reparameterize for holdout
            z, _ = model.reparameterize(mu, var, holdout_loader.dataset.y)
            z_inputs = utils.z_chess_aug_2k(z, holdout_loader.dataset.y, width=knot_width)
            z_origin = z_inputs.exp().reshape(z_inputs.shape[0], -1)
            y_star = model.softplus((interp_weight @ z_origin.T).add(interp_bias[:, None]))

            # Reparameterize for white noise
            z_white, _ = model.reparameterize(mu, var, white_noise_label)
            z_inputs_white = utils.z_chess_aug_2k(z_white, white_noise_label, width=knot_width)
            z_origin_white = z_inputs_white.exp().reshape(z_inputs_white.shape[0], -1)
            y_star_white = model.softplus((interp_weight @ z_origin_white.T).add(interp_bias[:, None]))

        # Select subset of rows (10:20)
        y_star = y_star[10:20, :]
        y_star_white = y_star_white[10:20, :]

        # Generate random error terms
        err_1 = torch.empty_like(y_star).exponential_(30)
        err_2 = torch.empty_like(y_star).bernoulli_(0.5) * 2 - 1

        # Generate emulated samples
        X_emu = (err_1 * err_2).exp() * y_star
        X_emu_white = (err_1 * err_2).exp() * y_star_white

        # Cast to float64 and store
        sample_1[:, :, i] = X_emu.to(torch.float32)
        sample_2[:, :, i] = X_emu_white.to(torch.float32)

        # Progress report
        if (i + 1) % 100 == 0 or i == n_samples - 1:
            print(f"Processed {i + 1}/{n_samples} samples")

    return sample_1, sample_2

def crps_boxplot(crps_list, labels, ax, log=False):
    """
    Plot CRPS results for multiple models as boxplots.

    Args:
        crps_list: list of 1D tensors or arrays of mean CRPS per location (e.g., [model1, model2, model3, model4])
        labels: list of strings, names corresponding to each model (e.g., ["Model 1", "Model 2", "Model 3", "Model 4"])
        ax: matplotlib axis
        log: bool, whether to apply log transform to CRPS values
    """
    data = [c.detach().cpu().numpy() for c in crps_list]

    if log:
        data = [np.log(np.clip(d, 1e-8, None)) for d in data]
        ax.set_ylabel("log(CRPS)", fontsize=14)
    else:
        ax.set_ylabel("CRPS", fontsize=14)

    colors = ["lightblue", "lightgreen", "lightcoral", "plum"][:len(data)]
    # ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        labels=labels,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="red", linewidth=2),
    )

    # Color each box individually
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":")

# Emulation
from scipy.interpolate import griddata

points = stations_train  # (2500, 2)
new_points = stations_holdout
white_noise_label = torch.randn_like(train_loader.dataset.y)
white_noise_label = (white_noise_label - white_noise_label.min()) / (white_noise_label.max() - white_noise_label.min())
white_noise_label = white_noise_label * MEI_scale


## CRPS

def compute_tail_weighted_crps(X_target, emulation, tail_quantile=0.9):
    """
    Compute tail-weighted CRPS using indicator weight I(z > 90th percentile of predictive samples).

    Args:
        X_target: Tensor of shape [n_loc, n_time] (true observations)
        emulation: Tensor of shape [n_loc, n_time, n_sample] (predictive samples)
        tail_quantile: float, e.g., 0.9 for top 10% tail

    Returns:
        CRPS: Tensor of shape [n_loc, n_time]
        CRPS_by_loc: mean CRPS per location
    """
    n_loc, n_time, n_sample = emulation.shape
    CRPS = torch.zeros(n_loc, n_time, device=emulation.device)

    for loc in range(n_loc):
        for t in range(n_time):
            samples = torch.sort(emulation[loc, t])[0]
            # Define tail threshold
            z_tail = samples[int(n_sample * tail_quantile)]

            # Integration grid from 0 to max(true, samples)
            upper_limit = torch.max(torch.cat([samples, X_target[loc, t].unsqueeze(0)])) + 1e-4
            z_grid = torch.linspace(0.0, upper_limit.item(), 1000, device=emulation.device)

            # Empirical CDF of predictive samples
            ecdf_vals = torch.tensor([torch.mean((samples <= z).float()) for z in z_grid], device=emulation.device)

            # Indicator of target <= z
            indicator = (X_target[loc, t] <= z_grid).float()

            # Tail weight: 1 if z > 90% quantile, else 0
            weight = (z_grid > z_tail).float()

            # Weighted integrand
            integrand = weight * (ecdf_vals - indicator) ** 2
            dz = z_grid[1] - z_grid[0]
            CRPS[loc, t] = torch.trapz(integrand, dx=dz)
            # Progress report
            if (t + 1) % 100 == 0 or t == n_time - 1:
                print(f"Processed {t + 1}/{n_time} times, Location {loc + 1}.")

    return CRPS

Emu_sample_model, Emu_sample_whitenoise = spatial_prediction(model, n_samples=2000)

start_time = time.time()
CRPS_model = compute_tail_weighted_crps(X_holdout, Emu_sample_model)
elapsed_time = time.time() - start_time
print(f"CRPS calculating time: {elapsed_time:.4f} seconds")

CRPS_white = compute_tail_weighted_crps(X_holdout, Emu_sample_whitenoise)

# Create the box plots
plt.figure(figsize=(6, 5))
plt.boxplot([CRPS_model.view(-1), CRPS_white.view(-1)], labels=["Model", "White noise"], patch_artist=True)

# Optional: make it look nicer
colors = ["skyblue", "lightcoral"]
for patch, color in zip(plt.gca().artists, colors):
    patch.set_facecolor(color)

plt.ylabel("Continuous Ranked Probability Score")
# plt.title("Comparison of MSE Between Two Models")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()

plt.figure(figsize=(6, 5))
data = [
    CRPS_model.view(-1),
    CRPS_white.view(-1)
]
# Create violin plot
parts = plt.violinplot(
    data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
)

# Set violin color
for pc in parts['bodies']:
    pc.set_facecolor("skyblue")
    pc.set_edgecolor("black")
    pc.set_alpha(0.8)

# Customize median lines
parts['cmedians'].set_color("darkblue")
parts['cmedians'].set_linewidth(1.5)

# X-axis labels
plt.xticks(
    [1, 2],
    ["Model", "White noise"]
)

plt.ylabel("Continuous Ranked Probability Score (CRPS)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Repeat for fixed W
# Bilinear interpolation
from scipy.interpolate import griddata

points = stations_train  # (2500, 2)

# New query points
new_points = stations_holdout

Emu_sample_model_fix, Emu_sample_whitenoise_fix = spatial_prediction(model, n_samples=2000)

start_time = time.time()
CRPS_model_fix = compute_tail_weighted_crps(X_holdout, Emu_sample_model_fix)
elapsed_time = time.time() - start_time
print(f"CRPS calculating time: {elapsed_time:.4f} seconds")

CRPS_white_fix = compute_tail_weighted_crps(X_holdout, Emu_sample_whitenoise_fix)

# Create the box plots
plt.figure(figsize=(8, 6))
bplot = plt.boxplot([CRPS_model.view(-1), CRPS_model_fix.view(-1),
                     CRPS_white.view(-1), CRPS_white_fix.view(-1)],
            labels=["Model", "Model*", "White noise", "White noise*"], patch_artist=True)

# Set all boxes to the same color
for patch in bplot['boxes']:
    patch.set_facecolor("skyblue")

plt.ylabel("Continuous Ranked Probability Score (CRPS)")
# plt.title("Comparison of MSE Between Two Models")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Create the box plots
plt.figure(figsize=(8, 6))
bplot = plt.boxplot([CRPS_model.log().view(-1), CRPS_model_fix.log().view(-1),
                     CRPS_white.log().view(-1), CRPS_white_fix.log().view(-1)],
            labels=["Model", "Model*", "White noise", "White noise*"], patch_artist=True)

# Set all boxes to the same color
for patch in bplot['boxes']:
    patch.set_facecolor("skyblue")

plt.ylabel("twCRPS (log scale)")
# plt.title("Comparison of MSE Between Two Models")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# QQ plots

qq_idx = 0
x = X_holdout[qq_idx, :].to(torch.float32)
y = Emu_sample_model[qq_idx, :, 0].to(torch.float32)
y_white = Emu_sample_whitenoise[qq_idx, :, 0].to(torch.float32)
y_fix = Emu_sample_model_fix[qq_idx, :, 0].to(torch.float32)
y_white_fix = Emu_sample_whitenoise_fix[qq_idx, :, 0].to(torch.float32)


qq_plot_quantile_scaled_custom(x, y, lim=1)
qq_plot_quantile_scaled_custom(x, y_white, lim=1)
qq_plot_quantile_scaled_custom(x, y_fix, lim=1)
qq_plot_quantile_scaled_custom(x, y_white_fix, lim=1)


X_emu = Emu_sample_model[:, :, 0]
X_emu_white = Emu_sample_whitenoise[:, :, 0]
X_emu_fix = Emu_sample_model_fix[:, :, 1]
X_emu_white_fix = Emu_sample_whitenoise_fix[:, :, 0]

MSE_model = torch.mean((X_emu - X_holdout).pow(2), dim=1)
MSE_white = torch.mean((X_emu_white - X_holdout).pow(2), dim=1)
MSE_model_fix = torch.mean((X_emu_fix - X_holdout).pow(2), dim=1)
MSE_white_fix = torch.mean((X_emu_white_fix - X_holdout).pow(2), dim=1)

# Create the box plots
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(8, 5))
plt.boxplot([MSE_model.view(-1), MSE_model_fix.view(-1),
             MSE_white.view(-1), MSE_white_fix.view(-1)],
            labels=["Model", "Model*", "White noise", "White noise*"], patch_artist=True)

# Optional: make it look nicer
colors = ["skyblue", "lightcoral"]
for patch, color in zip(plt.gca().artists, colors):
    patch.set_facecolor(color)

plt.ylabel("Mean Squared Error (MSE)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()



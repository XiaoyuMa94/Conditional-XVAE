import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

def softplus_clip(x, beta):
    x = x.clone()
    x = torch.where(x * beta > 700, 700 / beta, x)
    y = (1 / beta) * torch.log1p(torch.exp(beta * x))  # log1p is numerically stable
    return y

def x_aug(x):
    x_tmp = torch.zeros(x.shape[1], x.shape[0] * 3)
    for iter in range(x.shape[1]):
        if iter == 0:
            x_tmp[iter, :] = torch.cat([x[:, iter], x[:, iter], x[:, (iter + 1)]])
        elif iter == (x.shape[1] - 1):
            x_tmp[iter, :] = torch.cat([x[:, (iter - 1)], x[:, iter], x[:, iter]])
        else:
            x_tmp[iter, :] = torch.cat([x[:, (iter - 1)], x[:, iter], x[:, (iter + 1)]])
    return x_tmp

def aug_2k(dat, meis, rbf_vec, width):

    # Build the candidate input tensor for CNN with aug(log(data), mei) for each time
    dat_torch_enso = torch.zeros(dat.shape[1], 1, int(dat.shape[0]/width)*width*2)
    dat_torch_enso[:, :, 0::2] = dat.T.unsqueeze(1).log()
    dat_torch_enso[:, :, 1::2] = torch.repeat_interleave(meis, dat.shape[0]).reshape(-1, dat.shape[0]).unsqueeze(1)
    dat_torch_enso = dat_torch_enso.reshape(dat_torch_enso.shape[0], 1, -1, width*2)
    # Assign the corresponding entry to the 3-dimensional (neighborhood time) tensor for the CNN input
    # Still keep the 20 by 20 map size for the input
    dat_torch_enso_3d = torch.zeros(dat.shape[1], 3, int(dat.shape[0]/width), width*2)

    for iter in range(dat.shape[1]):
        if iter == 0:
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[(iter + 1), 0, :, :]
        elif iter == (dat.shape[1] - 1):
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[(iter - 1), 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[iter, 0, :, :]
        else:
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[(iter - 1), 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[(iter + 1), 0, :, :]

    # Build the data tensor for loss evaluation, the data is vec(data) here
    dat_extended = torch.cat([dat.log(), rbf_vec.repeat(dat.shape[1], 1).T, meis.T], dim=0)
    dat_3d = torch.zeros(dat.shape[1], 3, rbf_vec.shape[1] + dat.shape[0] + 1)
    for iter in range(dat.shape[1]):
        if iter == 0:
            dat_3d[iter, 0, :] = dat_extended[:, iter]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, (iter + 1)]
        elif iter == (dat.shape[1] - 1):
            dat_3d[iter, 0, :] = dat_extended[:, (iter - 1)]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, iter]
        else:
            dat_3d[iter, 0, :] = dat_extended[:, (iter - 1)]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, (iter + 1)]

    return dat_torch_enso_3d, dat_3d


def chessboard_aug(dat, meis, rbf_vec, width):
    # Define the indexes of chessboard
    odd_ind = np.array([])

    for i in range(dat.shape[0]):
        # width = np.sqrt(dat.shape[0])
        if ((i // width) % 2 == 0) and (i % 2 == 1):
            odd_ind = np.append(odd_ind, i)
        elif ((i // width) % 2 == 1) and (i % 2 == 0):
            odd_ind = np.append(odd_ind, i)

    # Build the candidate tensor with c(log(data), rbf_vec, mei) for each time
    dat_torch_enso = torch.zeros(dat.shape[1], 1, int(dat.shape[0]/width), width)

    for iter in range(dat.shape[1]):
        tmp = dat[:, iter].log()
        tmp[odd_ind] = meis[iter]
        tmp_mat = tmp.reshape(int(dat.shape[0]/width), width)
        dat_torch_enso[iter, 0, :, :] = tmp_mat

    # Assign the corresponding entry to the 3-dimensional (neighborhood time) tensor for the CNN input
    # Still keep the 20 by 20 map size for the input
    dat_torch_enso_3d = torch.zeros(dat.shape[1], 3, int(dat.shape[0]/width), width)

    for iter in range(dat.shape[1]):
        if iter == 0:
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[(iter + 1), 0, :, :]
        elif iter == (dat.shape[1] - 1):
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[(iter - 1), 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[iter, 0, :, :]
        else:
            dat_torch_enso_3d[iter, 0, :, :] = dat_torch_enso[(iter - 1), 0, :, :]
            dat_torch_enso_3d[iter, 1, :, :] = dat_torch_enso[iter, 0, :, :]
            dat_torch_enso_3d[iter, 2, :, :] = dat_torch_enso[(iter + 1), 0, :, :]

    # Build the data tensor for loss evaluation, the data is vec(data) here
    dat_extended = torch.cat([dat.log(), rbf_vec.repeat(dat.shape[1], 1).T, meis.T], dim=0)
    dat_3d = torch.zeros(dat.shape[1], 3, rbf_vec.shape[1] + dat.shape[0] + 1)
    for iter in range(dat.shape[1]):
        if iter == 0:
            dat_3d[iter, 0, :] = dat_extended[:, iter]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, (iter + 1)]
        elif iter == (dat.shape[1] - 1):
            dat_3d[iter, 0, :] = dat_extended[:, (iter - 1)]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, iter]
        else:
            dat_3d[iter, 0, :] = dat_extended[:, (iter - 1)]
            dat_3d[iter, 1, :] = dat_extended[:, iter]
            dat_3d[iter, 2, :] = dat_extended[:, (iter + 1)]

    return dat_torch_enso_3d, dat_3d

class CustomDataset(Dataset):
    def __init__(self, dat1, dat2):
        self.x = dat1
        self.y = dat2

    def __len__(self):
        ncol = self.x.shape[0]
        return ncol

    def __getitem__(self, idx):
        image = self.x[idx, :, :, :]
        out1 = self.y[idx, :, :]
        return image, out1


class CVAEinput_Dataset(Dataset):
    def __init__(self, dat1, dat2):
        self.x = dat1
        self.y = dat2
    def __len__(self):
        ncol = self.x.shape[0]
        return ncol

    def __getitem__(self, idx):
        image = self.x[idx, :]
        label = self.y[idx, :]
        return image, label


class CVAEinput_Dataset_gpu(Dataset):
    def __init__(self, dat1, dat2, device='cuda'):
        self.x = dat1
        self.y = dat2
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        image = self.x[idx, :].to(self.device)
        label = self.y[idx, :].to(self.device)
        return image, label


def wendland(d, r):
    """Wendland function with s=2, k=1."""
    out = ((1 - d / r) ** 4) * (4 * d / r + 1)
    out[d >= r] = 0  # zero beyond support
    return out

def z_chess_aug_2k(z, label, width):
    k = int(z.shape[1]/3)
    z = z.reshape(-1, 3, k)
    z_4d = torch.zeros([z.shape[0], 3, k*2])
    for iter in range(z.shape[1]):
        z_4d[:, iter, 0::2] = z[:, iter, :]
        z_4d[:, iter, 1::2] = label[:, iter].repeat_interleave(int(k)).reshape(z.shape[0], -1)
    z_4d = z_4d.reshape(z.shape[0], 3, -1, width*2)
    return z_4d

def generate_spatial_extremes(n_locations=500,
                              n_replicates=100,
                              range_param=0.2,
                              seed=42):
    """
    Generates spatial extreme value data with distance-based dependence

    Args:
        n_locations: Number of spatial locations
        n_replicates: Number of temporal replicates
        range_param: Range parameter for exponential covariance (controls dependence decay)
        seed: Random seed

    Returns:
        tuple: (Locations tensor, Data tensor)
    """
    torch.manual_seed(seed)

    # 1. Generate locations uniformly in [0,1]x[0,1]
    Loc = torch.rand(n_locations, 2)

    # 2. Compute pairwise distances
    Dist = torch.cdist(Loc, Loc)

    # 3. Create covariance matrix (exponential kernel)
    Cov = torch.exp(-Dist / range_param)

    # Add small diagonal for numerical stability
    Cov += torch.eye(n_locations) * 1e-6

    # 4. Cholesky decomposition
    try:
        L = torch.linalg.cholesky(Cov)
    except RuntimeError:
        # Handle non-positive definite matrix
        Cov += torch.eye(n_locations) * 1e-3
        L = torch.linalg.cholesky(Cov)

    # 5. Generate data
    Data = torch.zeros(n_replicates, n_locations)
    for i in range(n_replicates):
        # Generate standard normal variables
        z = torch.randn(n_locations)

        # Correlate through Cholesky
        y = L @ z

        # Transform to unit Fréchet margins
        Data[i, :] = -1 / torch.log(torch.special.ndtr(y))

    return Loc, Data.T  # Transpose to match (n_locations, n_replicates) shape



def ARE(u, unit_area, x):
    # s_0 = torch.randint(0, x.shape[0], (1,))
    # arbitrary location
    s_0 = torch.tensor([272])

    # Empirical marginal distribution functions
    # U_0r = empirical_cdf(x[s_0, :])
    U_ir = empirical_cdf_2d(x)

    ind_1, where_exceed = torch.where(U_ir[s_0, :] > u)
    if (where_exceed.shape[0] > 0):
        tmp_AE = torch.zeros(where_exceed.shape[0])
        for col in range(where_exceed.shape[0]):
            tmp_AE[col] = torch.sum(U_ir[:, where_exceed[col]] > u).mul(unit_area)

    truth = torch.median(tmp_AE).div(torch.pi).sqrt()
    tmp1, tmp2 = torch.quantile(tmp_AE, torch.tensor([0.025, 0.975]))
    truth_lower = truth + tmp1.div(torch.pi).sqrt().sub(truth).div(2)
    truth_upper = truth + tmp2.div(torch.pi).sqrt().sub(truth).div(2)

    # return {
    #     'truth': truth,
    #     'upper': truth_upper,
    #     'lower': truth_lower
    # }
    return torch.tensor([truth, truth_lower, truth_upper])


def calculate_and_plot_ARE(Data, unit_area=1, u_vec=None, n_reps=20, plot_ARE=False):

    # Repeat the ARE calculation for each replicate
    # u_vec = torch.cat([torch.arange(0, 0.99, 0.01),
    #                    torch.arange(0.9801, 0.9997, 0.0001)])
    ARE_res = torch.zeros((n_reps, 3, u_vec.shape[0]))

    for rep in range(n_reps):
        print(f"Processing replication {rep + 1}/{n_reps}")
        # ARE_res[rep] = torch.stack([ARE(u, unit_area, Data[rep, :, :]) for u in u_vec])
        ARE_truth = torch.zeros([3, u_vec.shape[0]])
        for iter in range(u_vec.shape[0]):
            ARE_truth[:, iter] = ARE(u_vec[iter], unit_area=unit_area, x=Data[rep, :, :])
        ARE_res[rep, :, :] = ARE_truth
    # Use mean as the final estimates
    ARE_mean = ARE_res[:, 0, :].mean(dim=0)
    low = ARE_res[:, 1, :].mean(dim=0)
    upp = ARE_res[:, 2, :].mean(dim=0)
    # low, upp = torch.quantile(ARE_res, torch.tensor([0.05, 0.95]), dim=0)

    # Plotting
    if plot_ARE:
        plt.figure(figsize=(10, 6))
        plt.plot(u_vec, ARE_mean, 'b-', label="ARE")
        plt.fill_between(u_vec, low, upp, color='lightblue', alpha=0.85, label="95% CI")
        plt.title("ARE Plot"), plt.xlabel("Quantile"), plt.ylabel("ARE")
        plt.legend(), plt.grid(True), plt.show()
    return {
        'truth': ARE_mean.numpy(),
        'upper': upp.numpy(),
        'lower': low.numpy()
    }

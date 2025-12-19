import netCDF4 as nc
import torch
import numpy as np
import os
import pandas as pd

os.chdir("E:/FWI/2014")

file_names = os.listdir()

# Open the NetCDF file
count = 1
final_dat = np.empty((245, 141, 145))

for iter in range(0, 245):
    print(f'Day: {count}')
    dat = nc.Dataset(file_names[iter], 'r')
    temp = dat.variables['GEOS-5_FWI']

    tmp_array = np.zeros((dat.variables['lat'][:].shape[0], dat.variables['lon'][:].shape[0]))

    for i in range(dat.variables['lat'][:].shape[0]):
        if i % 50 == 0:
            print(i)
        for j in range(dat.variables['lon'][:].shape[0]):
            tmp_array[i, j] = temp[0, i, j].data

    final_dat[iter, :, :] = tmp_array
    count = count + 1
timeline = np.linspace(1, 245, 245)

for i in range(245):
    ind = np.isnan(final_dat[i, 52:193, 928:1073]).sum()
    print(f'Number of NaN : {ind}')

dat_2014 = np.nan_to_num(final_dat, nan=-20)
dat_2014_tensor = torch.tensor(dat_2014)

non_nan_array = np.full((141, 145), np.nan)

for i in range(52, 193):
    print(i)
    for j in range(928, 1073):
        non_nan_array[i-52, j-928] = tmp1[0, i, j].data

for iter in range(final_dat.shape[0]):
    print(f'Day: {count}')
    dat = nc.Dataset(file_names[iter], 'r')
    temp = dat.variables['GEOS-5_FWI']

    tmp_array = np.full((141, 145), np.nan)

    for i in range(141):
        arr = non_nan_array[i, :]
        non_nan_indices = np.where(~np.isnan(arr))[0]

        if i % 50 == 0:
            print(i)
        tmp_array[i, non_nan_indices] = temp[0, i+52, non_nan_indices+928].data

    final_dat[iter, :, :] = tmp_array
    count = count + 1

dat_2014 = final_dat
dat_2014_tensor = torch.tensor(dat_2014)
torch.save(dat_2014_tensor, "E:/FWI/dat_2014_tensor.pt")

os.chdir("E:/FWI")
dat_2014 = torch.load("dat_2014_tensor.pt")
dat_2015 = torch.load("dat_2015_tensor.pt")
dat_2016 = torch.load("dat_2016_tensor.pt")
dat_2017 = torch.load("dat_2017_tensor.pt")
dat_2018 = torch.load("dat_2018_tensor.pt")
dat_2019 = torch.load("dat_2019_tensor.pt")
dat_2020 = torch.load("dat_2020_tensor.pt")
dat_2021 = torch.load("dat_2021_tensor.pt")
dat_2022 = torch.load("dat_2022_tensor.pt")
dat_2023 = torch.load("dat_2023_tensor.pt")
dat_2024 = torch.load("dat_2024_tensor.pt")

dat_all = torch.concat([dat_2014, dat_2015, dat_2016, dat_2017,
                        dat_2018, dat_2019, dat_2020, dat_2021,
                        dat_2022, dat_2023, dat_2024], dim=0)

dat_vec = dat_all.reshape(3897, -1)

for i in range(dat_vec.shape[1]):
    if i % 100 == 0:
        print(f"loc: {i}")
    for j in range(dat_vec.shape[0]):
        if torch.isnan(dat_vec[j, i]):
            dat_vec[j, i] = torch.mean(dat_vec[j-3:j+3, i][~torch.isnan(dat_vec[j-3:j+3, i])])

np.savetxt("dat_vec_fill.csv", dat_vec.numpy(), delimiter=',')
torch.save(dat_vec, "dat_fill_2_vec.pt")


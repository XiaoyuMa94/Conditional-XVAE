# =============================================================================
# data_preparation.R
#
# FWI data preparation pipeline for the real data analysis in Ma et al. (2025):
#   "Modeling Spatio-Temporal Extremes via Conditional Variational Autoencoders"
#
# This script processes raw daily FWI NetCDF files into model-ready inputs:
#   Step 1: Extract FWI data from NetCDF files for the target region
#   Step 2: Fill missing values via local temporal interpolation
#   Step 3: Compute monthly maxima
#   Step 4: Remove seasonality via cubic spline detrending (mgcv)
#   Step 5: Subset to eastern Australia domain
#   Step 6: Fit GEV distributions location-wise and transform margins
#   Step 7: Construct Wendland basis W and RBF basis for theta
#   Step 8: Export model inputs as CSV
#
# Raw FWI data source:
#   NASA NCCS Data Portal — https://portal.nccs.nasa.gov/datashare/GlobalFWI/
#   Files: FWI.GEOS-5.Daily.Default.YYYYMMDD.nc  (May 2014 - Nov 2024)
#
# ENSO index source:
#   NOAA Nina 3.4 index — https://psl.noaa.gov/data/climateindices/list/
#   File: Nina3.4.csv (provided in data/FWI/)
#
# Outputs (saved to DATA_DIR):
#   X_Data.csv     — GEV-transformed FWI (n_loc x n_t)
#   W_Data.csv     — Wendland basis matrix (n_loc x k)
#   RBF_Data.csv   — RBF basis for theta (k x n_basis)
#   MEIs_Data.csv  — normalized ENSO index (n_t x 1)
# =============================================================================

library(fields)
library(mgcv)
library(xts)
library(ismev)
library(ggplot2)
library(parallel)

# Set your data directory here
DATA_DIR <- "data/FWI"

# =============================================================================
# Helper Functions
# =============================================================================

softplus <- function(x, beta) {
  x[x * beta > 700] <- 700 / beta
  (1 / beta) * log(1 + exp(beta * x))
}

relu <- function(x) pmax(0, x)

wendland <- function(d, r) {
  # Wendland basis function (s=2, k=1)
  if (any(d < 0)) stop("d must be nonnegative")
  ((1 - d/r)^4 * (4 * d/r + 1)) * (d < r)
}

# =============================================================================
# Step 1: Extract FWI from NetCDF Files
# =============================================================================
# Raw NetCDF files are organized by year in separate subdirectories.
# For each year, loop over daily files and extract the target region:
#   Latitude:  52:193 (rows) -> -33.75 to -23.25 S  (eastern Australia)
#   Longitude: 928:1073 (cols) -> 143.125 to 150.9375 E
#
# NOTE: This step is computationally expensive. Run once per year and save
# the result as a .pt file to avoid re-processing.
#
# Example for one year (repeat for 2014-2024, adjusting the directory):
#
# library(netCDF4)  # or reticulate + netCDF4 Python package
# year_dir <- file.path(DATA_DIR, "raw", "2015")
# file_names <- list.files(year_dir, full.names = TRUE)
# n_days <- length(file_names)
# final_dat <- array(NA, dim = c(n_days, 141, 145))
#
# for (iter in seq_len(n_days)) {
#   dat <- nc_open(file_names[iter])
#   temp <- ncvar_get(dat, "GEOS-5_FWI")
#   final_dat[iter, , ] <- temp[928:1072, 52:192]   # lon x lat -> lat x lon
#   nc_close(dat)
#   if (iter %% 50 == 0) cat("Day:", iter, "/", n_days, "\n")
# }
# saveRDS(final_dat, file.path(DATA_DIR, "dat_2015.rds"))
# =============================================================================

# =============================================================================
# Step 2: Combine Years, Fill Missing Values, Compute Monthly Maxima
# =============================================================================

# Load pre-extracted yearly arrays (output from Step 1)
years <- 2014:2024
dat_list <- lapply(years, function(yr) {
  readRDS(file.path(DATA_DIR, paste0("dat_", yr, ".rds")))
})
dat_all <- do.call(rbind, lapply(dat_list, function(d) {
  matrix(d, nrow = dim(d)[1], ncol = prod(dim(d)[2:3]))
}))
# dat_all: (n_days x n_loc), n_days = 3897, n_loc = 141*145 = 20445

# Fill remaining NaN values with local temporal mean (window ±3 days)
cat("Filling missing values...\n")
for (i in seq_len(ncol(dat_all))) {
  if (i %% 500 == 0) cat("Location:", i, "/", ncol(dat_all), "\n")
  for (j in seq_len(nrow(dat_all))) {
    if (is.nan(dat_all[j, i])) {
      window <- dat_all[max(1, j-3):min(nrow(dat_all), j+3), i]
      dat_all[j, i] <- mean(window[!is.nan(window)], na.rm = TRUE)
    }
  }
}

# Compute monthly maxima (May 2014 to Nov 2024, n_month = 127)
date_range <- seq(as.Date("2014-05-01"), as.Date("2024-11-30"), by = "days")
stopifnot(length(date_range) == nrow(dat_all))

df_dates <- data.frame(Date = date_range, Index = seq_len(nrow(dat_all)))
df_dates$YearMonth <- format(df_dates$Date, "%Y-%m")

monthly_maxima <- do.call(rbind, lapply(split(df_dates, df_dates$YearMonth), function(grp) {
  apply(dat_all[grp$Index, , drop = FALSE], 2, max, na.rm = TRUE)
}))
# monthly_maxima: (127 x n_loc)

saveRDS(monthly_maxima, file.path(DATA_DIR, "monthly_maxima.rds"))
cat("Monthly maxima saved.\n")

# =============================================================================
# Step 3: Detrend Monthly Maxima via Cubic Spline (mgcv)
# =============================================================================
# Removes seasonality and long-term trend using a local pooling approach.
# For each location, a neighborhood within `radius` km is used to estimate
# the mean and standard deviation trends via:
#   - Mean: intercept + linear time + 12 cyclic cubic splines (seasonality)
#   - SD:   intercept + linear time (log-linear)
# Standardized residuals are stored in `residuals2`.
#
# WARNING: This step is computationally intensive. Use parallel processing.

monthly_maxima <- readRDS(file.path(DATA_DIR, "monthly_maxima.rds"))

# Full domain coordinates
lon_full <- seq(110, 155, 0.3125)   # 145 values
lat_full <- seq(-45, -10, 0.25)     # 141 values
loc_full <- cbind(
  rep(lon_full, length(lat_full)),
  rep(lat_full, each = length(lon_full))
)

# Transpose so dat is (n_time x n_loc) for detrending
dat <- t(monthly_maxima)   # (127 x 20445)
n_times <- nrow(dat)
n_sites <- ncol(dat)

# Build design matrix: intercept + linear time + 12 cyclic splines
day_of_year <- as.POSIXlt(date_range)$yday + 1
BS  <- cSplineDes(1:365, knots = quantile(1:365, (0:12)/12), ord = 4)
X_design <- cbind(1, (1:n_times) / (365 * 100))
for (i in seq_len(ncol(BS))) {
  X_design <- cbind(X_design, rep(BS[, i], length(unique(format(date_range, "%Y")))))
}

residuals2 <- matrix(NA, nrow = n_times, ncol = n_sites)
radius     <- 60   # km

detrend_one <- function(j) {
  ind_j    <- which(rdist.earth(matrix(loc_full[j,], ncol = 2),
                                loc_full, miles = FALSE) <= radius)
  data_ind <- dat[, ind_j, drop = FALSE]
  valid    <- complete.cases(t(data_ind))
  data_j_mat <- data_ind[, valid, drop = FALSE]

  if (sum(data_j_mat, na.rm = TRUE) == 0) return(rep(NA, n_times))

  data_j <- as.vector(data_j_mat)
  X_j    <- do.call(rbind, replicate(ncol(data_j_mat), X_design, simplify = FALSE))

  fit_mean <- lm(data_j ~ X_j - 1)
  est_mean_j <- fit_mean$fitted.values[
    nrow(X_design) * (which(ind_j == j) - 1) + seq_len(nrow(X_design))
  ]
  resid1_j <- dat[, j] - est_mean_j

  # Log-linear SD model
  X_sd <- X_j[, 1:2]
  nllk <- function(param) {
    sds <- exp(X_sd %*% param)
    sum(log(sds) + 0.5 * ((data_j - fit_mean$fitted.values) / sds)^2)
  }
  fit_sd <- optim(c(sd(resid1_j), 0), nllk, method = "Nelder-Mead",
                  control = list(maxit = 1000))
  est_sd_j <- exp(X_design[, 1:2] %*% fit_sd$par)

  resid1_j / est_sd_j
}

# Run in parallel (Windows: use makeCluster; Mac/Linux: use mclapply)
n_cores <- max(1, detectCores() - 2)
cl <- makeCluster(n_cores)
clusterEvalQ(cl, { library(fields); library(mgcv) })
clusterExport(cl, c("loc_full", "radius", "dat", "X_design"), envir = environment())

cat("Detrending", n_sites, "locations using", n_cores, "cores...\n")
res <- parLapply(cl, seq_len(n_sites), detrend_one)
stopCluster(cl)

for (j in seq_len(n_sites)) residuals2[, j] <- res[[j]]

saveRDS(residuals2, file.path(DATA_DIR, "residuals2.rds"))
cat("Detrending complete.\n")

# =============================================================================
# Step 4: Subset to Eastern Australia and Compute Monthly Maxima of Residuals
# =============================================================================

residuals2 <- readRDS(file.path(DATA_DIR, "residuals2.rds"))

# Eastern Australia: lat 45:87 (rows), lon 106:131 (cols) within full domain
# Corresponds to 143.125-150.9375 E, 33.75-23.25 S
east_aus_residuals <- residuals2[,
  which(loc_full[, 1] >= 143.125 & loc_full[, 1] <= 150.9375 &
        loc_full[, 2] >= -33.75  & loc_full[, 2] <= -23.25)
]
# east_aus_residuals: (127 x 1118)

# These are already monthly — no further aggregation needed
dat_east <- t(east_aus_residuals)   # (1118 x 127)
dat_east[dat_east == -Inf] <- -20   # replace -Inf from log(0) cases

# Eastern Australia coordinates
lon_east <- seq(143.125, 150.9375, 0.3125)   # 26 values
lat_east <- seq(-33.75,  -23.25,   0.25)     # 43 values
new_loc  <- cbind(
  rep(lon_east, length(lat_east)),
  rep(lat_east, each = length(lon_east))
)
stations <- data.frame(x = new_loc[, 1], y = new_loc[, 2])

# =============================================================================
# Step 5: Fit GEV Distributions and Transform Margins
# =============================================================================
# Fit GEV location-wise and apply the marginal transformation from Appendix 4.
# The transformation maps FWI onto the positive support needed by the cXVAE.

library(ismev)

n_loc_east <- nrow(dat_east)
fitted_gev_par <- data.frame(matrix(NA, nrow = n_loc_east, ncol = 3))
colnames(fitted_gev_par) <- c("location", "scale", "shape")

cat("Fitting GEV distributions...\n")
for (i in seq_len(n_loc_east)) {
  gev_fit <- tryCatch(
    gev.fit(dat_east[i, ], show = FALSE)$mle,
    error = function(e) rep(NA, 3)
  )
  fitted_gev_par[i, ] <- gev_fit
  if (i %% 100 == 0) cat("Location:", i, "/", n_loc_east, "\n")
}

save(fitted_gev_par, file = file.path(DATA_DIR, "fitted_gev_par.RData"))

# GEV marginal transformation: map X -> positive support
# Based on the upper endpoint parameterization (Appendix 4 of paper)
fitted_gev_par$beta <- fitted_gev_par$location -
  fitted_gev_par$scale / fitted_gev_par$shape

dat_marginalized <- matrix(NA, nrow = n_loc_east, ncol = ncol(dat_east))

for (iter in seq_len(n_loc_east)) {
  beta_tmp <- fitted_gev_par$beta[iter]
  tau_tmp  <- fitted_gev_par$scale[iter]
  xi_tmp   <- fitted_gev_par$shape[iter]

  if (is.na(xi_tmp)) next

  if (xi_tmp > 0) {
    dat_marginalized[iter, ] <-
      (xi_tmp * (dat_east[iter, ] - beta_tmp) / tau_tmp)^(1 / xi_tmp)
  } else {
    dat_marginalized[iter, ] <-
      (tau_tmp / (-xi_tmp * (beta_tmp - dat_east[iter, ])))^(1 / abs(xi_tmp))
  }

  if (iter %% 100 == 0) cat("Transforming location:", iter, "\n")
}

X <- dat_marginalized

# =============================================================================
# Step 6: ENSO Index
# =============================================================================

nina34 <- as.matrix(read.table(
  file.path(DATA_DIR, "nina34.txt"), row.names = 1, quote = "\""
))
nina34 <- nina34[65:75, ]                        # 2014-2024
nina34_vec <- c(t(nina34))[5:131]                # 127 monthly values

# Normalize to [0.02, 0.98]
nina_tmp <- 0.02 + (nina34_vec - min(nina34_vec)) /
  (max(nina34_vec) - min(nina34_vec)) * (0.98 - 0.02)

# =============================================================================
# Step 7: Wendland Basis W and RBF Basis for Theta
# =============================================================================

knot <- expand.grid(
  x = seq(min(new_loc[, 1]), max(new_loc[, 1]), 0.4),
  y = seq(min(new_loc[, 2]), max(new_loc[, 2]), 0.4)
)

k   <- nrow(knot)
eucD <- rdist(stations, as.matrix(knot))
W    <- wendland(eucD, r = 0.9)
W    <- sweep(W, 1, rowSums(W), FUN = "/")   # row-normalize

# RBF basis for tilting parameter field theta
rbf_basis <- function(x, y, center, tau_sq = 2) {
  exp(-((x - center[1])^2 + (y - center[2])^2) / tau_sq)
}

center_grid <- expand.grid(
  x = seq(min(new_loc[, 1]) + 0.1, max(new_loc[, 1]), 1),
  y = seq(min(new_loc[, 2]) + 0.1, max(new_loc[, 2]), 1)
)

rbf_mat <- matrix(NA, nrow = nrow(knot), ncol = nrow(center_grid))
for (i in seq_len(nrow(center_grid))) {
  rbf_mat[, i] <- with(knot, rbf_basis(x, y, center_grid[i, ]))
}
rbf_mat <- sweep(rbf_mat, 1, rowSums(rbf_mat), FUN = "/")   # row-normalize

# =============================================================================
# Step 8: Export Model Inputs
# =============================================================================

# Set output directory (modify as needed)
# setwd("path/to/output/directory")

write.csv(X,        file.path(DATA_DIR, "X_Data.csv"),    row.names = FALSE)
write.csv(W,        file.path(DATA_DIR, "W_Data.csv"),    row.names = FALSE)
write.csv(rbf_mat,  file.path(DATA_DIR, "RBF_Data.csv"),  row.names = FALSE)
write.csv(nina_tmp, file.path(DATA_DIR, "MEIs_Data.csv"), row.names = FALSE)

cat("All model inputs saved to", DATA_DIR, "\n")
cat("X shape:       ", nrow(X),       "x", ncol(X),       "\n")
cat("W shape:       ", nrow(W),       "x", ncol(W),       "\n")
cat("RBF shape:     ", nrow(rbf_mat), "x", ncol(rbf_mat), "\n")
cat("ENSO length:   ", length(nina_tmp), "\n")

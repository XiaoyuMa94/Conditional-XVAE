# =============================================================================
# simulate_data.R
#
# Generates spatio-temporal extreme fields conditioned on the ENSO index
# for the simulation study in Ma et al. (2025):
#   "Modeling Spatio-Temporal Extremes via Conditional Variational Autoencoders"
#
# The simulation follows the max-id model in Section 3 of the paper:
#   X(s) = epsilon(s) * Y(s)
# where epsilon(s) ~ log-Laplace(0, 1/alpha0) and
#   Y(s) = sum_k omega_k(s)^(1/alpha) * Z_k,  Z_k ~ expPS(alpha, theta_k(c_t))
#
# Inputs:
#   Nina3.4.csv  — raw ENSO index (available from NOAA:
#                  https://psl.noaa.gov/data/climateindices/list/)
#
# Outputs (saved as CSV in the working directory):
#   X_Data.csv      — observed spatial fields       (n.s x n.t)
#   Y_Data.csv      — latent Y process              (n.s x n.t)
#   Z_Data.csv      — latent expPS variables        (k x n.t)
#   Thetas_Data.csv — tilting parameters            (k x n.t)
#   W_Data.csv      — Wendland basis matrix         (n.s x k)
#   RBF_Data.csv    — RBF basis matrix for theta    (k x 144)
#   MEIs_MA_Data.csv — smoothed ENSO index          (n.t x 1)
# =============================================================================

library(fields)
library(VGAM)
library(zoo)

# =============================================================================
# Helper Functions
# =============================================================================

relu <- function(x) pmax(0, x)

wendland <- function(d, r) {
  # Wendland basis function (s=2, k=1)
  if (any(d < 0)) stop("d must be nonnegative")
  return(((1 - d/r)^4 * (4 * d/r + 1)) * (d < r))
}

# Rejection sampler for expPS(alpha=0.5, theta) latent variables
# Based on Bopp, Shaby and Huser (2021)
single_rejection_sampler <- function(theta) {
  X <- invgamma::rinvgamma(1, shape = 1/2, scale = 4)
  V <- runif(1)
  while (V > exp(-theta * X)) {
    X <- invgamma::rinvgamma(1, shape = 1/2, scale = 4)
    V <- runif(1)
  }
  return(X)
}

# Tilting parameter field theta_t(c_t) as a function of ENSO index
# Center shifts along the off-diagonal from (20,0) to (0,20) as c_t increases
Theta_surf <- function(mei, phi = 12, knot, b = 2, var = 0.05, c1 = 1, c2 = 1) {
  center   <- mei * c(0, 20) + (1 - mei) * c(20, 0)
  distance <- sqrt(((knot[,1] - center[1])^2) / c1 +
                     ((knot[,2] - center[2])^2) / c2)
  C <- var * exp(-(distance / phi)^b)
  return(var - C)
}

# =============================================================================
# ENSO Index: Load, Subset, Smooth, and Normalize
# =============================================================================

nina34   <- read.csv("Nina3.4.csv")
nina34   <- as.matrix(nina34[31:74, 2:13])   # 1980-2023, 44 years x 12 months
MEIs     <- c(t(nina34))                      # n.t = 528 monthly values

# Normalize to [0.01, 0.99]
MEIs_norm <- (MEIs - min(MEIs)) * 0.98 / (max(MEIs) - min(MEIs)) + 0.01

# 5-month centered moving average for smoother representation
MEIs_MA        <- rollapply(MEIs_norm, width = 5, FUN = mean,
                            align = "center", fill = NA)
MEIs_MA[1:2]   <- c(mean(MEIs_norm[1:3]), mean(MEIs_norm[1:4]))
MEIs_MA[527:528] <- c(mean(MEIs_norm[525:528]), mean(MEIs_norm[526:528]))

# =============================================================================
# Spatial Setup: Grid, Knots, and Wendland Basis
# =============================================================================

set.seed(123)

# 50x50 regular grid over [0,20]x[0,20] — n.s = 2500 locations
stations <- data.frame(expand.grid(
  x = seq(0, 20, length = 50),
  y = seq(0, 20, length = 50)
))

# 15x15 knot grid — K = 225 knots
knot <- expand.grid(
  x = seq(0.5, 19.5, length.out = 15),
  y = seq(0.5, 19.5, length.out = 15)
)

k   <- nrow(knot)       # 225
n.s <- nrow(stations)   # 2500
n.t <- length(MEIs_MA)  # 528

# Wendland basis matrix W: shape (n.s x k), row-normalized
eucD    <- rdist(stations, as.matrix(knot))
W       <- wendland(eucD, r = 3)
W       <- sweep(W, 1, rowSums(W), FUN = "/")
W_alpha <- W^(1/0.5)   # alpha = 0.5

# =============================================================================
# RBF Basis for Tilting Parameter Theta
# =============================================================================

rbf <- function(x, y, center, rho = 1, tau_sq = 100) {
  d       <- sqrt((x - as.numeric(center[1]))^2 + (y - as.numeric(center[2]))^2)
  dist_sq <- exp(-(d / tau_sq)^rho)
  return(dist_sq)
}

center_grid <- expand.grid(
  x = seq(0.2, 19.8, length.out = 12),
  y = seq(0.5, 19.8, length.out = 12)
)

rbf_mat <- matrix(NA, nrow = nrow(knot), ncol = nrow(center_grid))
for (i in 1:nrow(center_grid)) {
  rbf_mat[, i] <- with(knot, rbf(x, y, center_grid[i,], tau_sq = 8, rho = 2))
}
rbf_mat <- rbf_mat / max(rbf_mat) * 0.05   # scale to match theta range

# =============================================================================
# Simulate Tilting Parameters Theta_t
# =============================================================================

alpha  <- 0.5
Thetas <- matrix(NA, nrow = k, ncol = n.t)

for (iter in 1:n.t) {
  Thetas[, iter] <- Theta_surf(MEIs_MA[iter], knot = knot, phi = 8, b = 2, var = 0.05)
}

# =============================================================================
# Simulate Latent Variables Z and Spatial Process X
# =============================================================================

set.seed(15)

Z               <- matrix(NA, nrow = k,   ncol = n.t)
X               <- matrix(NA, nrow = n.s, ncol = n.t)
Epsilon         <- matrix(NA, nrow = n.s, ncol = n.t)
Y               <- matrix(NA, nrow = n.s, ncol = n.t)

alpha0 <- 30  # log-Laplace scale parameter

for (iter in 1:n.t) {
  # Sample latent expPS variables knot-wise
  for (i in 1:k) {
    Z[i, iter] <- single_rejection_sampler(theta = Thetas[i, iter])
  }
  
  # log-Laplace noise: epsilon = exp(Laplace(0, 1/alpha0))
  half_exp_tmp    <- rexp(n.s, rate = alpha0) * (2 * rbinom(n.s, size = 1, prob = 0.5) - 1)
  Epsilon[, iter] <- exp(half_exp_tmp)
  
  # Low-rank spatial process Y and observed field X
  Y[, iter] <- W_alpha %*% Z[, iter]
  X[, iter] <- Epsilon[, iter] * Y[, iter]
  
  if (iter %% 50 == 0) cat("Simulated time point", iter, "/", n.t, "\n")
}

# =============================================================================
# Save Outputs
# =============================================================================

# Set output directory (modify as needed)
# setwd("path/to/output/directory")

write.csv(X,        "X_Data.csv",       row.names = FALSE)
write.csv(Y,        "Y_Data.csv",       row.names = FALSE)
write.csv(Z,        "Z_Data.csv",       row.names = FALSE)
write.csv(Thetas,   "Thetas_Data.csv",  row.names = FALSE)
write.csv(W,        "W_Data.csv",       row.names = FALSE)
write.csv(rbf_mat,  "RBF_Data.csv",     row.names = FALSE)
write.csv(MEIs_MA,  "MEIs_MA_Data.csv", row.names = FALSE)

cat("Simulation complete. All outputs saved.\n")
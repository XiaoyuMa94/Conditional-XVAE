source("utils.R")
cols <- rev(RColorBrewer::brewer.pal(n=9, "RdBu"))

#### ENSO ####
nina34 <- read.csv("Nina3.4.csv") 
nina34 <- as.matrix(nina34[31:74, 2:13]) # 1980-2023
MEIs <- c(t(nina34))

MEIs_norm <- (MEIs-min(MEIs)) * (0.98) / (max(MEIs)-min(MEIs)) + 0.01
MEIs_MA <- rollapply(MEIs_norm, width = 5, FUN = mean, align = "center", fill = NA)
MEIs_MA[1:2] <- c(mean(MEIs_norm[1:3]),mean(MEIs_norm[1:4]))
MEIs_MA[527:528] <- c(mean(MEIs_norm[525:528]),mean(MEIs_norm[526:528]))

#### Simulation ####
alpha = 0.5

##### Generate knots #####
set.seed(123)

stations <- data.frame(expand.grid(x=seq(0,20,length=50),y=seq(0,20,length=50)))

knot <- expand.grid(x=seq(0.5,19.5, length.out=8), y=seq(0.5,19.5, length.out=8))

k = nrow(knot)
n.s <- nrow(stations)
n.t <- length(MEIs)

##### Basis construction (Wendland basis) #####

eucD <- rdist(stations,as.matrix(knot))
W <- wendland(eucD,r=8)
W <- sweep(W, 1, rowSums(W), FUN="/")

##### Basis construction (radial basis) #####

rbf <- function(x, y, center,rho = 1,tau_sq=100) {
  tmp <- (sqrt((x-as.numeric(center[1]))^2 + (y-as.numeric(center[2]))^2)/tau_sq)^rho
  dist_sq <- exp(-tmp)
  return(dist_sq)
}
center_grid <- expand.grid(x=seq(0,20,length.out=5),y=seq(0,20,length.out=5))

rbf_mat <- matrix(NA,ncol = nrow(center_grid),nrow = nrow(knot))
for (i in 1:nrow(center_grid)) {
  rbf_mat[,i] <- with(knot,rbf(x,y,center_grid[i,],tau_sq=5,rho = 2))
}
rbf_mat <- sweep(rbf_mat,1,rowSums(rbf_mat),FUN="/")

##### Simulate tilting parameters theta, latent Z and response X #####

Thetas <- matrix(NA, nrow = k, ncol = n.t)
Z <- matrix(NA, nrow=k, ncol=n.t)
X <- matrix(NA, nrow=n.s, ncol=n.t)
Epsilon_frechet <- matrix(NA, nrow=n.s, ncol=n.t)
y <- matrix(NA, n.s, n.t)

for (iter in 1:n.t) {
  Thetas[,iter] <- Theta_surf(MEIs_MA[iter], knot = knot, phi = 15, var = 2, b = 2)
}

set.seed(123)
for (iter in 1:n.t) {
  for (i in 1:k) {
    Z[i,iter] <- single_rejection_sampler(theta = Thetas[i,iter])
  }
  half_exp_tmp <- rexp(n.s, rate=30)*(2*rbinom(n.s,size=1,prob=0.5)-1)
  Epsilon_frechet[,iter] <-  exp(half_exp_tmp)
  y[, iter] <- ((W_alpha)%*%Z[,iter])
  X[,iter] <-  Epsilon_frechet[,iter]* (y[, iter])
  print(iter)
}

##### Save the data #####
setwd("~/Data")

write.csv(W,"W_Data.csv")
write.csv(rbf_mat,"RBF_Data.csv")
write.csv(y,"Y_Data.csv")
write.csv(X,"X_Data.csv")
write.csv(Z,"Z_Data.csv")
write.csv(Thetas,"Thetas_Data.csv")
write.csv(MEIs_MA,"MEIs_MA_Data.csv")

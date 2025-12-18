library(fields)
library(VGAM)
library(ggplot2)
library(torch)
library(dplyr)
library(fields)
library(zoo)
library(ggforce)
library(ggpubr)

Theta_surf <- function(mei, phi=12,knot=knot,b=2,var=0.05, c1=1, c2=1){
  center <- mei*c(1,20)+(1-mei)*c(20,1)
  distance <- sqrt(((knot[,1]-center[1])^2)/c1 + ((knot[,2]-center[2])^2)/c2)
  
  C <- var*exp(-(distance/phi)^b)
  return(var-C)
}

softplus <- function(x, beta){
  x[x*beta>700] = 700/beta
  y = (1/beta) * log(1+exp(beta*x))
  # y[is.infinite(y)] = 1000
  return(y)
}
relu <- function(x){
  return(pmax(0,x))
  # return(log(1+exp(x)))
}
wendland <- function (d,r) {
  if (any(d < 0)) 
    stop("d must be nonnegative")
  # return((1 - d/r)^2 * (d < r)) # s = 2; k = 0
  return(((1 - d/r)^4 * (4 * d/r + 1)) * (d < r)) # s = 2; k = 1
  # return(((1 - d/r)^6 * (35 * (d/r)^2 + 18 * (d/r) + 3)) * (d < r)) # s = 2; k = 2
}


# alpha is fixed at 1/2
single_rejection_sampler = function(theta=theta){
  X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
  V <- runif(1)
  while(V> exp(-theta*X)){
    X <- invgamma::rinvgamma(1, shape=1/2, scale=4)
    V <- runif(1)
  }
  return(X)
}

# alpha is not necessarily 1/2
single_rejection_sampler_alpha_not_half = function(theta=theta, alpha){
  gamma <- cos(pi*alpha/2)^{1/alpha}
  X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
  V <- runif(1)
  while(V> exp(-theta*X)){
    X <- stabledist::rstable(1, alpha=alpha, beta = 1, gamma = gamma, delta = 0, pm=1)
    V <- runif(1)
  }
  return(X)
}


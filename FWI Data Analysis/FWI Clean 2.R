
##### data #####

##### time (day/month/year) #####

# # Define the start and end dates
start_date <- as.POSIXlt("2014-05-01")
end_date <- as.POSIXlt("2024-11-30")

# Generate a sequence of dates from start to end
dates_sequence <- seq(from = start_date, to = end_date, by = "days")
# Convert the Date object to POSIXlt
time <- as.POSIXlt(dates_sequence)

day <- time$mday
month <- time$mon + 1
year <- time$year + 1900

##### loc #####
lon <- seq(110, 155, 0.3125)
lat <- seq(-45, -10, 0.25)
loc <- cbind(rep(lon, length(lat)), rep(lat, each=length(lon)))

setwd("E:/FWI/")
data = read.csv("dat_vec_fill.csv", header = FALSE)
FWI_mat = as.matrix(data)
save(FWI_mat, time, loc, day, month, year, file = "FWI_daily.Rdata")

#### REPEAT Red_Sea_Temperature_Application ####
library(parallel)
library(fields)
library(mgcv)
library(ggplot2)
# library(tictoc)

load("FWI_daily.Rdata") ## Loads the original dataset
data <- FWI_mat[1:3867,]
save(data, file = "FWI_daily.RData")
x <- unique(loc[,1]) ## Longitudes
y <- unique(loc[,2]) ## Latitudes

nx <- length(x) ## number of longitude points
ny <- length(y) ## number of latitude points

n.times <- nrow(data) ## number of time points
n.sites <- ncol(data) ## number of spatial sites
n.years <- length(unique(year)) ## number of years


#######################
### MARGINAL TRENDS ###
#######################

BS <- cSplineDes(c(1:365),knots=quantile(c(1:365),c(0:12)/12),ord=4) ## basis of 12 cyclic cubic splines (one per month) for capturing seasonality

par(mfrow=c(3,4), mar = c(3, 3, 2, 2))
for(i in 1:ncol(BS)){
  plot(c(1:365),BS[,i],type="l", xlab = "Days", ylab = "Value")
}

## to see the chosen knots: attr(BS,"knots")

## Creation of the design matrix (intercept,linear time, 12 cyclic splines)
X <- cbind(1,c(1:n.times)/(365*100))
for(i in 1:ncol(BS)){
  X <- cbind(X,rep(BS[,i],n.years))
}

n.par <- ncol(X) ## number of covariates / regression parameters

## Creation of R objects that contain the resulting fits 
# est.par.mean <- matrix(nrow=n.par,ncol=n.sites)
# est.mean <- matrix(nrow=n.times,ncol=n.sites)
# residuals1 <- matrix(nrow=n.times,ncol=n.sites)
# est.par.sd <- matrix(nrow=2,ncol=n.sites)
# est.sd <- matrix(nrow=n.times,ncol=n.sites)
residuals2 <- matrix(nrow=n.times,ncol=n.sites)

radius <- 60 ##in km (defines the local neighborhoods for estimating trends)

FUN <- function(j){
  # print(j)
  ind.j <- which(rdist.earth(x1=matrix(loc[j,],ncol=2),x2=loc,miles=FALSE)<=radius)
  data.ind <- data[,ind.j]
  non_NA_ind <- complete.cases(t(data.ind))
  data.j.mat <- data.ind[,non_NA_ind]
  if(sum(data.j.mat)!=0){
    
    data.j <- as.vector(data.j.mat)
    
    X.j <- X
    for(i in 1:(ncol(data.j.mat)-1)){
      X.j <- rbind(X.j,X)
    }
    X.j.mean <- as.matrix(X.j) #for modeling the mean, we use the intercept, time trend, and 12 cyclic splines for seasonality
    X.j.sd <- as.matrix(X.j[,1:2]) #for modeling the standard deviation, we only use the intercept and time trend
    
    ## Linear model for estimating the mean structure
    fit.mean.j <- lm(data.j~X.j.mean-1)
    res1.est.par.mean <- fit.mean.j$coefficients
    est.mean.j <- fit.mean.j$fitted.values
    residuals1.j <- data.j-est.mean.j
    res2.est.mean <- est.mean.j[nrow(X)*(which(ind.j==j)-1) + c(1:nrow(X))]
    res3.residuals1 <- data[,j]-res2.est.mean
    
    ## negative log-likelihood for estimating the standard deviation
    nllk.j <- function(param,residuals1.j,X.j.sd){
      sds <- exp(X.j.sd%*%param)
      return(sum(log(sds)+0.5*(residuals1.j/sds)^2))
    }
    init <- c(sd(res3.residuals1),0)
    fit.sd.j <- optim(par=init,fn=nllk.j,residuals1.j=residuals1.j,
                      X.j.sd=X.j.sd,method="Nelder-Mead",
                      control=list(maxit=1000),hessian=FALSE)
    
    res4.est.par.sd <- fit.sd.j$par
    est.sd.j <- exp(X.j.sd%*%res4.est.par.sd)
    res5.est.sd <- exp(X[,1:2]%*%res4.est.par.sd)
    # residuals2.j <- residuals1.j/est.sd.j
    res6.residuals2 <- res3.residuals1/res5.est.sd
    # return(list(res6.residuals2))
    return(list(res1.est.par.mean,res2.est.mean,res3.residuals1,
                res4.est.par.sd,res5.est.sd,res6.residuals2))
  } else {
    return(list(matrix(NA, length(time), ncol = 1)))
  }

}


for(j in 1:n.sites){
  est.par.mean[,j] <- res[[j]][[1]]
  est.mean[,j] <- res[[j]][[2]]
  residuals1[,j] <- res[[j]][[3]]
  est.par.sd[,j] <- res[[j]][[4]]
  est.sd[,j] <- res[[j]][[5]]
  residuals2[,j] <- res[[j]][[6]]
  print(j)
}

save(list=c("BS","X","radius","est.par.mean","est.mean","residuals1","est.par.sd","est.sd","residuals2"),file="Trends_fits.RData")
load(file="Trends_fits.RData")


save(loc,file="location.RData")
save(data,X,file="data&X.RData")
save(est.mean,est.par.mean,est.par.sd,est.sd,file="est.RData")
save(residuals1,residuals2,file="residuals.RData")
save(time,file="time.RData")


#### After detrending ####
dat <- residuals2

# Define the start and end dates
start_date <- as.POSIXlt("2014-05-01")
end_date <- as.POSIXlt("2024-11-30")

# Generate a sequence of dates from start to end
dates_sequence <- seq(from = start_date, to = end_date, by = "days")
# Convert the Date object to POSIXlt
time <- as.POSIXlt(dates_sequence)
n_month <- 127
library(xts)

mon_max_allsites <- matrix(NA,nrow = ncol(dat),ncol = n_month) 
for (i in 1:ncol(dat)) {
  xts.ts <- xts(dat[,i],time)
  ## Get monthly maxima
  mon_max_allsites[i,] <- as.numeric(apply.monthly(xts.ts, function(x) max(x, na.rm = TRUE)))
  if(i%%100==0) cat("row: ",i, "\n")
}

write.csv(mon_max_allsites,"FWI_monmax_detrended.csv")

##### Select the target region #####
dat <- read.csv("FWI_monmax_detrended.csv", header = TRUE)
dat <- as.matrix(dat)[, 2:128]

lon_temp <- dataset$var$lon[929:1073]   
lat_temp <- dataset$var$lat[53:193]     

east_aus_dat <- dat[, 46:88, 107:132]
east_aus_dat <- matrix(east_aus_dat, nrow = 127)

# Save to CSV
write.table(
  east_aus_dat,
  file = "East_Aus_dat.csv",
  sep = ",",
  row.names = FALSE,
  col.names = FALSE
)

#### GOF ####
library(ismev)
library(gnFit)
library(EnvStats)
library(distributional)
library(fitdistrplus)
library(dplyr)
library(stringr)
library(ggplot2)
library(gridExtra)
setwd("E:/FWI/")

lon <- seq(143.125, 150.9375, 0.3125)
lat <- seq(-33.75, -23.25, 0.25)
new_loc <- cbind(rep(lon, length(lat)), rep(lat, each = length(lon)))

dat = read.csv("East_Aus_dat.csv", header = FALSE)
dat <- t(as.matrix(dat))


cut_point <- quantile(dat,probs = seq(0, 1, 0.05))
n_int <- length(cut_point)-1

oi_vec <- matrix(NA,ncol = 20,nrow = nrow(dat))
ei_vec <- matrix(NA,ncol = 20,nrow = nrow(dat))

#### GEV distribution (GOF) ####
fitted_gev_par <- data.frame(matrix(NA,nrow = nrow(dat),ncol = 4))
colnames(fitted_gev_par) <- c("location","scale","shape","p-val")
for (i in 1:nrow(dat)) {
  gev_par <- gev.fit(dat[i,],show = FALSE)$mle
  oi <- table(cut(dat[i,], breaks = cut_point)) %>% as.numeric()
  oi_vec[i,] <- oi
  oi[oi==0] <- NA
  ei <- diff(extRemes::pevd(cut_point[1:21],loc=gev_par[1],scale = gev_par[2],shape = gev_par[3],
                            type = "GEV"))*127
  ei_vec[i,] <- ei
  chisq_stat <- sum(na.omit(oi*log(oi/ei)))
  chisq_pval <- pchisq(chisq_stat,df=n_int-4,lower.tail = FALSE)
  fitted_gev_par[i,] <- c(gev_par,chisq_pval)
  if(i%%10==0) cat("row: ",i, "\n")
}

#### T distribution (GOF) ####

cut_point <- quantile(dat,probs = seq(0, 1, 0.05))
n_int <- length(cut_point)-1

fitted_t_res <- matrix(NA,nrow = nrow(dat),ncol = 4)
for (i in 1:nrow(dat)) {
  
  t_par <- try(fitdistr(dat[i,],"t"),silent = TRUE)
  if("try-error" %in% class(t_par)) {
    fitted_t_res[i,] <- rep(NA,4)
  } else {
    oi <- table(cut(dat[i,], breaks = cut_point)) %>% as.numeric()
    oi[oi==0] <- NA
    t_dist <- dist_student_t(df = t_par$estimate[3], mu = t_par$estimate[1], sigma = t_par$estimate[2])
    ei <- diff(cdf(t_dist,cut_point)[[1]])*127
    chisq_stat <- sum(na.omit(oi*log(oi/ei)))
    p_val <- pchisq(chisq_stat,df=n_int-4,lower.tail = FALSE)
    
    fitted_t_res[i,] <- c(t_par$estimate,p_val)
  }
  if(i%%100==0) cat("row: ",i, "\n")
}

#### The marginal of transformed varible does not match the marginal distr in our model
#### So transform it one more time with the upper bound of GEV

dat_marginalized <- matrix(NA,nrow = nrow(dat),ncol = ncol(dat))

fitted_gev_par$beta <- fitted_gev_par$location - fitted_gev_par$scale/fitted_gev_par$shape
for(iter in 1:nrow(dat_marginalized)){
  beta_tmp <- fitted_gev_par$beta[iter]
  tau_tmp <- fitted_gev_par$scale[iter]
  xi_tmp <-  fitted_gev_par$shape[iter]
  xi_cons <- 1
  if (xi_tmp>0){
    dat_marginalized[iter, ] <- ((xi_tmp * (dat[iter,] - beta_tmp))/tau_tmp)^{1/(xi_tmp*xi_cons)}
  } else {
    dat_marginalized[iter, ] <- (tau_tmp/(-(xi_tmp)*(beta_tmp - dat[iter,])))^{1/abs(xi_tmp*xi_cons)}
  }
  
  if(iter%%100==0) cat("Location: ",iter,"\n")
}

X <- dat_marginalized

##### ENSO #####
# Read nina 3.4
nina34 <- as.matrix(read.table("~/nina34.txt", row.names=1, quote="\""))
nina34 <- nina34[c(65:75),]
nina34_vec <- c(t(as.matrix(nina34)))[5:131]

nina_tmp <- 0.02+(nina34_vec-min(nina34_vec))/(max(nina34_vec)-min(nina34_vec))*(0.98-0.02)

##### W #####

knot <- expand.grid(x=seq(min(new_loc[,1]),max(new_loc[,1]), 0.4), 
                    y=seq(min(new_loc[,2]),max(new_loc[,2]), 0.4))

k = nrow(knot)
n.s <- nrow(stations)
n.t <- length(nina_tmp) # n.t <- 500

eucD <- rdist(stations,as.matrix(knot))

W <- wendland(eucD,r=0.9)
W <- sweep(W, 1, rowSums(W), FUN="/")

#### RBF ####

rbf <- function(x, y, center,theta_coef=1,tau_sq=100) {
  dist_sq <- theta_coef*exp(-((x-as.numeric(center[1]))^2 + (y-as.numeric(center[2]))^2)/tau_sq)
  return(dist_sq)
}
center_grid <- expand.grid(x=seq(min(new_loc[,1])+0.1,max(new_loc[,1]),1),
                           y=seq(min(new_loc[,2])+0.1,max(new_loc[,2]),1))

rbf_mat <- matrix(NA,ncol = nrow(center_grid),nrow = nrow(knot))
for (i in 1:nrow(center_grid)) {
  rbf_mat[,i] <- with(knot,rbf(x,y,center_grid[i,],tau_sq=2))
}
rbf_mat <- sweep(rbf_mat,1,rowSums(rbf_mat),FUN="/")

##### Save #####
setwd("C:/Users/User/OneDrive - University of Missouri/VAE Project/CVAE/py code/Real Data/FWI")

write.csv(X,"X_Data.csv")
write.csv(W,"W_Data.csv")
write.csv(rbf_mat,"RBF_Data.csv")
write.csv(nina_tmp,"MEIs_Data.csv")

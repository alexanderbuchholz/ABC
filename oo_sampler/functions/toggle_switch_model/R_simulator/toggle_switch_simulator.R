library(truncnorm)
robservation <- function(nobservations, theta){
  # constants used in the data generating process
  h <- 1
  tau <- 300
  u0 <- 10
  v0 <- 10
  #
  u <- matrix(0, nrow = nobservations, ncol = tau+1)
  v <- matrix(0, nrow = nobservations, ncol = tau+1)
  u[,1] <- u0
  v[,1] <- v0
  # noise <- array(randomness[(nobservations+1):length(randomness)], dim = c(nobservations, tau, 2))
  for (t in 1:tau){
    u[,t+1] <- u[,t] + h * theta[1] / (1 + v[,t]^theta[3]) - h * (1 + 0.03 * u[,t])
    u[,t+1] <- u[,t+1] + h * 0.5 * rtruncnorm(nobservations, a = - u[,t+1]/(h*0.5))
    v[,t+1] <- v[,t] + h * theta[2] / (1 + u[,t]^theta[4]) - h * (1 + 0.03 * v[,t])
    v[,t+1] <- v[,t+1] + h * 0.5 * rtruncnorm(nobservations, a = - v[,t+1]/(h*0.5))
  }
  y <- u[,tau+1] + theta[5] + theta[5] * theta[6] * rnorm(nobservations) / (u[,tau+1]^theta[7])
  return(y)
}
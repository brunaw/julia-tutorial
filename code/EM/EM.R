library(tidyverse)

samples <- read_rds("code/EM/samples.rds")
attach(samples) # data, pi0, mu0, sigma0

# Plotting -----
data %>% 
  as.data.frame() %>% 
  ggplot(aes(x = X1, X2)) +
  geom_point() +
  theme_minimal()

# ------------------------------------------------
X = data 
mu = matrix(c(mu0), ncol = 2, byrow = TRUE)
pi = pi0
sigma = sigma0
# ------------------------------------------------
E_step <- function(X, pi, mu, sigma, C = 3){
  N = nrow(X)
  d = ncol(X)
  gamma = matrix(rep(NA, times = N * C), ncol = C)
  x = matrix(rep(NA, times = N * d), ncol = d)
  
  for(c in 1:C) {
    cst = ((2 * 3.141592)^d * det(sigma[, , c]))^0.5
    
    for(i in 1:d) x[, i] = X[ ,i] - mu[c, i]
    xt = t(x)
    xinvSxt = x %*% solve(sigma[, , c]) * x
    xinvSx = rowSums(xinvSxt)*  -1/2
    dens = exp(xinvSx)/cst
    dens[1]
    gamma[, c] = pi[c] * dens
  }
  
  gamma_div = rowSums(gamma)
  final_gamma = gamma / gamma_div
  return(final_gamma)
}

M_step <- function(X, gamma){
  N = nrow(X)
  C = ncol(gamma)
  d = ncol(X)
  mu = matrix(rep(NA, times = C * d), ncol = d)
  sigma = array(NA, c(d, d, C))
  gamma_sum = colSums(gamma)
  pi = gamma_sum/N
  x = matrix(rep(NA, times = N * d), ncol = d)
  for(c in 1:C){
    for (m in 1:d) {
      mu[c, m] = sum(gamma[, c]* X[, m])/gamma_sum[c]
    }
    
    for(i in 1:d){
      x[, i] = X[ ,i] - mu[c, i]
    }
    
    sigma[, , c] = (t(x) %*% diag(gamma[, c]) %*% x)/gamma_sum[c]
  }
  return(list(pi = pi, mu = mu, sigma = sigma))
  
}

compute_vlb <- function(X, pi, mu, sigma, gamma, C = 3){
  N = nrow(X)
  d = ncol(X)
  loss = 0
  
  for(c in 1:C){
    for(n in 1:N) {
      logd = emdbook::dmvnorm(
        x = X[n, ], mu = c(mu[c,]), Sigma = sigma[, , c], 
        log = TRUE
      )
      
      gamma_c = gamma[n, c]
      loss_here = (gamma_c * ((log(pi[c]) + logd) -
                                log(gamma_c)))
      loss = loss + loss_here
      }
  }
  return(loss)
}


gamma <-  E_step(X, pi, mu, sigma)
m <-  M_step(X, gamma)
vlb <-  compute_vlb(X, m$pi, m$mu, m$sigma, gamma)

train_EM <- function(X, C, rtol = 1e-3, 
                     max_iter = 100, restarts = 10){
  N = nrow(X)
  d = ncol(X)
  best_loss = -Inf
  best_pi = NA
  best_mu = NA
  best_sigma = NA

  for(i in 1:restarts){
    print(i)
    pi = runif(C)
    pi = pi/sum(pi)
    mu = matrix(rnorm(C * d), nrow = C)
    sigma = array(0, c(d, d, C))
    for(c in 1:C){
      diag(sigma[, , c]) <- 1
    }
    lloss = -Inf
    not_saturated = TRUE
    iteration = 0
    #while (not_saturated & (iteration < max_iter)){
    for(j in 1:max_iter){
      print(j)
      former_loss = lloss
      gamma <-  E_step(X, pi, mu, sigma)
      m <-  M_step(X, gamma)
      pi <- m$pi
      mu <- m$mu
      sigma <- m$sigma
      lloss = compute_vlb(X, pi, mu, sigma, gamma)
      #if(former_loss > lloss) print("bug")
      not_saturated = abs((lloss/former_loss) - 1) > rtol
      #iteration = iteration + 1
    }
    if(lloss > best_loss){
      best_loss = lloss
      best_pi = m$pi
      best_mu = m$mu
      best_sigma = m$sigma
    }
  }
  return(list(best_loss = best_loss, 
              best_pi = best_pi,
              best_mu = best_mu, 
              best_sigma = best_sigma))
}


EM <- train_EM(X, 3)
gamma = E_step(X, EM$best_pi, EM$best_mu, EM$best_sigma)

gamma %>% 
  as.data.frame() %>% 
  gather() %>% 
  mutate(obs = rep(1:280, times = 3)) %>% 
  group_by(obs) %>% 
  filter(value == max(value)) %>% 
  arrange(obs) %>% 
  bind_cols(as.data.frame(X)) %>% 
  ggplot(aes(x = X1, X2)) +
  geom_point(aes(colour = key)) +
  theme_minimal()

# best_pi = pi
# best_mu = mu
# best_sigma = sigma
# end
# end
# return best_loss, best_pi, best_mu, best_sigma
# end





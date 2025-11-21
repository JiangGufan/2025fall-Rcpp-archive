# 使用benchmark测速
library(microbenchmark)

mean_forloop = function(x){
  n = length(x)
  s = 0
  for(i in 1:n){
    s = s + x[i]
  }
  return(s/n)
}

mean_direct = function(x){
  return(mean(x))
}

p = 1e4
xData = runif(p)
testResult = microbenchmark(mean_forloop(xData),
                            mean_direct(xData), times = 50)

# > testResult
# Unit: microseconds
# expr     min      lq     mean  median      uq      max neval
# mean_forloop(xData) 119.187 119.269 146.7308 119.515 121.278 1345.333    50
# mean_direct(xData)  11.726  11.849  19.1921  11.972  12.136  365.925    50

#  eg04 do vector convolution
convolve_r <-function(x,y){
  stopifnot(length(x)==length(y))
  n <- length(x); s=0
  for(i in seq_len(n)) s <- s + x[i]*y[n-i+1]
  s
}

simple_r <- function(x,y) sum(x * rev(y))
Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg04_conv.cpp")
x <- 1:5; y <- 1:5
convolve_r(x, y)
convolve_c(x, y)
simple_r(x, y)

library(microbenchmark)
set.seed(1); p <- 1e4; xr <- runif(p); yr <- runif(p)
microbenchmark(convolve_c(xr,yr),convolve_r(xr,yr),simple_r(xr,yr),times=50)

# Unit: microseconds
# expr     min      lq      mean   median      uq     max neval
# convolve_c(xr, yr)   5.248   5.453  10.42712   5.5965   5.781 244.770    50
# convolve_r(xr, yr) 250.674 251.904 257.76864 256.4345 257.398 308.115    50
# simple_r(xr, yr)  30.217  31.447  44.00448  32.3490  33.497 582.856    50

#
Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg05_localSmooth.cpp")
set.seed(1); y <- rnorm(10)
w <- c(1,2,1); w <- w / sum(w)       # 高斯样式的简单核
cbind(y, localSmoothing(y, w))

# y
# [1,] -0.6264538 -0.22140524
# [2,]  0.1836433 -0.27369894
# [3,] -0.8356286  0.02691673
# [4,]  1.5952808  0.67111019
# [5,]  0.3295078  0.35845699
# [6,] -0.8204684 -0.20599999
# [7,]  0.4874291  0.22317861
# [8,]  0.7383247  0.63496495
# [9,]  0.5757814  0.39612476
# [10,] -0.3053884  0.13519648


Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg06_matmult.cpp")
library(microbenchmark); set.seed(1)
p <- 200; r <- 500; q <- 1000
A <- matrix(rnorm(p*r), p, r)
B <- matrix(rnorm(r*q), r, q)
microbenchmark(matrixMult_v1(A,B), matrixMult_v2(A,B), matrixMult_v3(A,B), times=10)

# Unit: milliseconds
# expr      min       lq     mean   median       uq      max neval
# matrixMult_v1(A, B) 62.02226 62.34706 62.91104 62.89990 63.14111 64.02253    10
# matrixMult_v2(A, B) 30.45931 30.59461 30.92410 30.78032 31.14688 31.75458    10
# matrixMult_v3(A, B) 25.29696 25.52299 25.68048 25.57631 25.80204 26.42704    10


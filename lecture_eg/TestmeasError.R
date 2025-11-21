library(measError)
set.seed(100)
n = 400
beta0 = 1
beta1 = 2
sigmaU = runif(n, 0.4, 0.6)
sigmaQ = runif(n, 0.1, 0.2)
sigmaE = 0.2
simuData = getData(n, beta0, beta1, sigmaU, sigmaQ, sigmaE)
(betaHat1 = naiveRegression(simuData$D, simuData$W))
(betaHat2 = correctedRegression(simuData$D, simuData$W, sigmaU))

library(tidyverse)
ggplot(simuData, aes(W, D)) + geom_point() + theme_bw() + 
  geom_abline(slope = betaHat1[1], intercept = betaHat1[2], color = "red") +
  geom_abline(slope = betaHat2[1], intercept = betaHat2[2], color = "blue") +
  geom_abline(slope = beta0, intercept = beta1)
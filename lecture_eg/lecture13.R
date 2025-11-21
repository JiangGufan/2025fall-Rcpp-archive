library(foreach)
x = foreach(i=1:3) %do% sqrt(i)

set.seed(100)
x <- foreach(i=1:4, .combine = 'cbind') %do% rnorm(4)
x


library(doSNOW)
numCores = parallel::detectCores()
cl<-makeCluster(numCores)
registerDoSNOW(cl)
result = foreach(i=1:4) %dopar% sqrt(i)
stopCluster(cl)


library(doParallel)
cl <- makeCluster(4) ## or cl <- makeForkCluster(4)
registerDoParallel(cl)
foreach(i=1:3) %dopar% sqrt(i)
stopCluster(cl)








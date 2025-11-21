library(Rcpp)
library(RcppArmadillo)

# eg07 fastLM
linearRegression = function(y,X){
  nSample = nrow(X)
  pCovariate = ncol(X)
  
  XtX = t(X) %*% X
  XtY = t(X) %*% y
  betaHat = solve(XtX,XtY)
  
  residuals = y-X %*% betaHat
  sigmaHat = sum(residuals*residuals) / (nSample- pCovariate)
  sigmaHat = sqrt(sigmaHat)
  betaSigma = sigmaHat *sqrt(diag(solve(XtX)))
  
  result = list(betaHat=betaHat,
                betaSigma=betaSigma,
                sigmaHat=sigmaHat)
  return(result)
}
Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg07_fastLM.cpp")
set.seed(42)
n <- 5000
p <- 5
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))  # 含截距项
beta_true <- 1:p
y <- X %*% beta_true + rnorm(n)

res_R <- linearRegression(y, X)
res_Cpp <- fastLM(y, X)

cat("✅ β估计比较:\n")
print(cbind(R = round(res_R$betaHat, 4),Cpp = round(res_Cpp$beta, 4)))
cat("\n✅ σ估计比较:\n")
print(c(R = res_R$sigmaHat, Cpp = res_Cpp$sigma))

library(microbenchmark)
mb <- microbenchmark(R_version = linearRegression(y, X),Cpp_version = fastLM(y, X),times = 20)
print(mb)

# 矩阵求逆数值不稳定
D1 = diag(c(1e2,1e-2))
D2 = diag(c(1e2,2e-2))
U = matrix(c(1,1,-1,1), nrow = 2)
A = t(U) %*% D1 %*% U
B = t(U) %*% D2 %*% U
# > A
# [,1]   [,2]
# [1,] 100.01 -99.99
# [2,] -99.99 100.01
# > B
# [,1]   [,2]
# [1,] 100.02 -99.98
# [2,] -99.98 100.02
# > solve(A)
# [,1]    [,2]
# [1,] 25.0025 24.9975
# [2,] 24.9975 25.0025
# > solve(B)
# [,1]    [,2]
# [1,] 12.5025 12.4975
# [2,] 12.4975 12.5025


# 求逆矩阵？
Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg08_InverseOrSolve.cpp")
set.seed(100)
library(Matrix)
n = 7
A = as.matrix(Hilbert(n))
eigen(A)$value
x0 = rnorm(n)
b = A %*% x0
computeInverse(A,b,x0)
SolveLS(A,b,x0)


library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg09_Decompose.cpp")
set.seed(100)
## 构造 SPD 矩阵（可控制条件数）
make_spd <- function(n, kappa = 1e3){
  Q <- qr.Q(qr(matrix(rnorm(n*n), n, n)))  # 随机正交
  s <- exp(seq(0, log(kappa), length.out = n))  # 单调正数奇异值
  A <- Q %*% diag(s) %*% t(Q)             # SPD: Q diag(s) Qᵀ
  A
}

set.seed(42)
n <- 800
A  <- make_spd(n, kappa = 1e6)            # 改 kappa 看病态度对误差/速度的影响
x_true <- rnorm(n)
b  <- A %*% x_true                        # 无噪声；可加微小噪声测试鲁棒性

# 误差（||x-x_true||2, ||Ax-b||2）
errs <- rbind(
  solve = t(SolveLS(A, b, x_true)),
  lu    = t(LU(A, b, x_true)),
  chol  = t(Cholesky(A, b, x_true)),
  ldl   = t(LDL(A, b, x_true)),
  svd   = t(SVD(A, b, x_true))
)
colnames(errs) <- c("x_err_2norm", "residual_2norm")
print(round(errs, 6))

# 运行时间（毫秒）
mb <- microbenchmark(
  SolveLS(A, b, x_true),
  LU(A, b, x_true),
  Cholesky(A, b, x_true),
  LDL(A, b, x_true),
  SVD(A, b, x_true),
  times = 10L
)
mb
# Unit: milliseconds
# expr       min        lq      mean    median        uq       max neval
# SolveLS(A, b, x_true)  44.12473  44.21120  45.69929  44.86763  47.56119  48.38426    10
# LU(A, b, x_true)  45.23825  45.47933  46.64420  46.28533  47.90276  48.81407    10
# Cholesky(A, b, x_true)  48.86249  49.55408  49.93859  49.67950  49.89536  52.93522    10
# LDL(A, b, x_true)  59.57878  60.23556  61.11811  60.38970  61.95932  65.05831    10
# SVD(A, b, x_true) 670.00359 670.89657 675.22114 673.36038 677.09725 686.17338    10
# solve结果好，他是高度优化的智能封装，它会自动检测矩阵类型（对称、正定、稀疏等）

Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg10_lognorm.cpp")
set.seed(1)
d <- 4
mu <- rnorm(d)
# 构造对称正定协方差
A <- matrix(rnorm(d*d), d, d)
Sigma <- crossprod(A) + diag(0.5, d)  # SPD
x <- rnorm(d)
# 计算对数密度
logpdf_mvn(x, mu, Sigma)
# [1] -10.61319
library(mvtnorm)
dmvnorm(x, mean = mu, sigma = Sigma, log = TRUE)
# [1] -10.61319


Rcpp::sourceCpp("~/MY_RUC/Rcoding/Rcpp/eg11_moreONreg.cpp")
set.seed(42)  
symSqrt(100)
# $relErr_method1
# [1] 1.459317e-15
# $relErr_method2
# [1] 1.511661e-15

out <- subview_index()
str(out$A)                 # 5x10 矩阵
str(out$X)                 # 5x5 矩阵
length(out$picked_gt_0p5)  # 每次长度可能不同
out$X                      # 看 2,3,6,8 位置是否被置为1（列主序）
# > str(out$A)                 # 5x10 矩阵
# num [1:5, 1:10] 0 0 0 0 0 ...
# > str(out$X)                 # 5x5 矩阵
# num [1:5, 1:5] 0.215 0.733 1 1 0.229 ...
# > length(out$picked_gt_0p5)  # 每次长度可能不同
# [1] 13
# > out$X
# [,1]      [,2]       [,3]
# [1,] 0.2152317 0.7702120 0.04442815
# [2,] 0.7334972 1.0000000 0.20506265
# [3,] 1.0000000 0.7295571 0.25032158
# [4,] 1.0000000 1.0000000 0.81521844
# [5,] 0.2289389 0.2768114 0.33370450
# [,4]      [,5]
# [1,] 0.82153720 0.4769173
# [2,] 0.89396046 0.7786021
# [3,] 0.05789266 0.1098504
# [4,] 0.81161949 0.7862721
# [5,] 0.17237826 0.6084200

set.seed(1)
n <- 200; p <- 50
X <- matrix(rnorm(n*p), n, p)
beta_true <- c(runif(10, -2, 2), rep(0, p-10))
y <- X %*% beta_true + rnorm(n, sd = 2)

# 指数网格：覆盖宽一些
lambdas <- exp(seq(log(1e-6), log(1e3), length.out = 100))

fit <- ridge_loocv_svd(X, as.vector(y), lambdas)

str(fit$lambda_best)
coef_best <- c(fit$b0_best, fit$beta_best)  # 截距 + 系数

# 1) 选最优 λ
k <- which.min(fit$loocv)
lam <- lambdas[k]
# 2) 取最优系数（当前函数是“无截距、无标准化”的岭解）
beta_hat <- fit$Beta[, k]
# 3) 用闭式 (X'X + λI)^{-1} X'y 校验（同样是无截距版本）
p <- ncol(X)
beta_closed <- solve(crossprod(X) + lam * diag(p), crossprod(X, y))
# 4) 两者应当极接近
max_abs_diff <- max(abs(beta_hat - beta_closed))
max_abs_diff
# [1] 2.248202e-15



# 计算多元正态分布的对数概率密度，R version
lognorm_r <- function(x, mu, Sigma){
  d <- length(mu)
  logdet_Sigma <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)
  inv_Sigma <- solve(Sigma)
  diff <- x - mu
  quad_form <- t(diff) %*% inv_Sigma %*% diff
  logpdf <- -0.5 * (d * log(2 * pi) + logdet_Sigma + quad_form)
  return(as.numeric(logpdf))
}

# 测试 R 版和 C++ 版结果一致性
set.seed(123)
d <- 6
mu <- rnorm(d)
A <- matrix(rnorm(d*d), d, d)
Sigma <- crossprod(A) + diag(0.1, d)  # SPD
x <- rnorm(d)
logpdf_R <- lognorm_r(x, mu, Sigma)
logpdf_Cpp <- logpdf_mvn(x, mu, Sigma)
cat("R 版对数密度:", logpdf_R, "\n")
cat("C++ 版对数密度:", logpdf_Cpp, "\n")


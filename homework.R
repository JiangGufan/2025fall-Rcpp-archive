
# Q1:利用RcppEigen重写Lecture notes 11中的fixedRankApprox 算法，实现图片压缩

Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/fixedRankApproxEigen.cpp")
library(jpeg)

# 读入图片并取灰度通道
img <- readJPEG("/Users/jiang/MY_RUC/Rcoding/Rcpp/HW3img.jpg")
img <- img[,,1]

# 低秩近似（不同 K）
img2 <- fixedRankApproxEigen(img, 5)
img3 <- fixedRankApproxEigen(img, 20)
img4 <- fixedRankApproxEigen(img, 50)

# 画原图 + 三个压缩版本
par(mar = c(0,0,0,0))
plot(0, 0, type = "n", xlim = c(-1, 1), ylim = c(-1, 1), axes = FALSE, xlab = "", ylab = "")

rasterImage(img,  -1,  0,  0,  1)  # 原图
rasterImage(img2,  0,  0,  1,  1)  # K = 5
rasterImage(img3, -1, -1,  0,  0)  # K = 20
rasterImage(img4,  0, -1,  1,  0)  # K = 50


# Q2:完成Lecture notes 11最后power iteration的三个task
# Task 1:用Armadillo实现power iteration，迭代一千次，计算照片的最大奇异值、奇异向量
library(Rcpp)
library(jpeg)
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/power_arma.cpp")
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/power_eigen.cpp")
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/power_eigen_sparse.cpp")

# 读灰度图（例如 9000 x 6000）
img <- readJPEG("/Users/jiang/MY_RUC/Rcoding/Rcpp/HW3img.jpg")   # 可以换成下载的图片路径
if (length(dim(img)) == 3L) img <- img[,,1]  # 只取一个通道变灰度

# 用于稀疏矩阵
m <- nrow(img); n <- ncol(img)
N <- m * n

set.seed(123)

# Task 1: Armadillo power iteration
time_arma <- system.time(
  res_arma <- power_iter_arma(img, n_iter = 1000)
)

time_arma
res_arma$sigma1   # 最大奇异值
res_arma$u1
res_arma$v1 #为左右奇异向量


# Task 2: 将照片矩阵随机系数化，用RcppEigen实现power iteration，迭代一千次。比较Armadillo代码和RcppEigen代码的运算速度

# 随机系数化（例如乘上一个 N(0,1) 的随机矩阵）
rand_mat <- matrix(rnorm(length(img)), nrow = nrow(img))
img_rand <- img * rand_mat

# Armadillo
t_arma <- system.time(
  res_arma_rand <- power_iter_arma(img_rand, n_iter = 1000)
)

# Eigen (dense)
t_eigen <- system.time(
  res_eigen_rand <- power_iter_eigen_dense(img_rand, n_iter = 1000)
)

t_arma
t_eigen


# Task 3: 比较在不同稀疏度(抽样个数)下的运算速度和精确度

## 1. 先用完整矩阵得到“真”的 v1，用来做精度评估
full_res <- power_iter_arma(img, n_iter = 1000)
v1_full  <- full_res$v1  # 长度 n

## 2. 定义一系列稀疏度（抽样个数 n_sample）
n_samples_grid <- as.integer(c(1e4, 5e4, 1e5, 3e5, 5e5, 8e5, 1e6))  # 可按自己机器调

times   <- numeric(length(n_samples_grid))
errors  <- numeric(length(n_samples_grid))

for (k in seq_along(n_samples_grid)) {
  n_sample <- n_samples_grid[k]
  n_sample <- min(n_sample, N)
  
  # 随机抽样 n_sample 个像素位置（不放回）
  idx <- sample.int(N, n_sample)
  # 把一维 index 转成 (row, col)
  row_idx <- (idx - 1L) %% m        # 0-based
  col_idx <- (idx - 1L) %/% m       # 0-based
  values  <- img[cbind(row_idx + 1L, col_idx + 1L)]
  
  # 计时：在稀疏矩阵上做 power iteration
  t_sparse <- system.time(
    res_sparse <- power_iter_eigen_sparse(
      i = row_idx,
      j = col_idx,
      x = values,
      m = m, n = n,
      n_iter = 1000
    )
  )
  
  times[k] <- t_sparse["elapsed"]
  
  v1_hat <- res_sparse$v1
  
  # 奇异向量符号不唯一，因此取 min(||v1_hat - v1||, ||v1_hat + v1||)
  diff1 <- sqrt(sum((v1_hat - v1_full)^2))
  diff2 <- sqrt(sum((v1_hat + v1_full)^2))
  errors[k] <- min(diff1, diff2)
  
  cat("n_sample =", n_sample,
      "time =", times[k],
      "error =", errors[k], "\n")
}

## 3. 作图：n (稀疏度) vs 运行时间
plot(n_samples_grid, times, type = "b", log = "x",
     xlab = "Number of sampled entries (n)",
     ylab = "Elapsed time (seconds)",
     main = "Runtime vs Sparsity (power iteration, 1000 iters)")

## 4. 作图：n (稀疏度) vs 精度 ||v1_hat -/+ v1||
plot(n_samples_grid, errors, type = "b", log = "x",
     xlab = "Number of sampled entries (n)",
     ylab = "Error ||v1_hat -/+ v1||_2",
     main = "Accuracy vs Sparsity (largest right singular vector)")


# Q3：尝试运行Lecture notes 12中的高斯过程插值代码，并展示结果
library(Rcpp)
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/Gaussian_Interpolation.cpp")

D = 5000
n = 100
sigmaSq_y = 0.1

sj = seq(0, 1, length.out = D)
x = sin(6*pi*sj)

jSet = sample(1:D, n, replace = FALSE)
yVec = x[jSet] + rnorm(n, sd = sqrt(sigmaSq_y))

#microbenchmark::microbenchmark(gpFitting(jSet - 1, yVec, 1e7, sigmaSq_y, D), times = 10)
fHat = gpFitting(jSet - 1, yVec, 1e8, sigmaSq_y, D)

plot(sj[jSet], yVec, pch = 20, cex = 0.5, type = "n")
polygon(c(sj, rev(sj)), 
        c(fHat[,1]+fHat[,2], rev(fHat[,1]-fHat[,2])),
        col = "lightgrey", border = "white")
lines(sj,x, col = "black", lwd = 2)
lines(sj, fHat[,1], col = "red", lwd = 2)
points(sj[jSet], yVec, col = "black", pch = 20, cex = 0.5)


# Q4:利用OpenMP实现 Image Convolution
library(Rcpp)
library(jpeg)
library(microbenchmark)
Rcpp::sourceCpp("/Users/jiang/MY_RUC/Rcoding/Rcpp/openMP_ImgConv.cpp")

# 读图，取灰度
img <- readJPEG("/Users/jiang/MY_RUC/Rcoding/Rcpp/HW3img.jpg")
if (length(dim(img)) == 3L) img <- img[,,1]

img_mat <- as.matrix(img)

# # 高斯平滑核
# list of matrix kernels
smoothM = matrix(c(1,2,1,2,4,2,1,2,1)/16, 3,3)
sharpenM = matrix(c(0,-1,0,-1,5,-1,0,-1,0),3,3)
edgeM = matrix(c(-1,-1,-1,-1,8,-1,-1,-1,-1),3,3)

# 跑一次看是否正常
res_serial  <- imageConv_serial(img_mat,  smoothM)
res_parallel <- imageConv_parallel(img_mat, smoothM, nthreads = 8)

# 对比运行时间
microbenchmark(
  imageConv_parallel(img_mat, smoothM, nthreads = 8),
  imageConv_serial(img_mat,  smoothM),
  times = 5
)



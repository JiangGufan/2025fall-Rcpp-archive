// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// 使用 Eigen 的命名空间
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;

// [[Rcpp::export]]
Rcpp::NumericMatrix fixedRankApproxEigen(const Rcpp::NumericMatrix &A, int K) {
  // 尺寸
  int m = A.nrow();
  int n = A.ncol();
  
  // 把 R 的 NumericMatrix 映射为 Eigen::MatrixXd（列主序和 Eigen 一致）
  Eigen::Map<const MatrixXd> M(A.begin(), m, n);
  
  // 计算薄 SVD：A = U * diag(s) * V^T
  JacobiSVD<MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  MatrixXd U = svd.matrixU();
  MatrixXd V = svd.matrixV();
  VectorXd s = svd.singularValues();
  
  int r = s.size();
  if (K < 0) K = 0;
  if (K > r) K = r;
  
  // 将第 K 个之后的奇异值置零（相当于保留前 K 个奇异值）
  for (int i = K; i < r; ++i) {
    s(i) = 0.0;
  }
  
  // 重构近似矩阵：U * diag(s) * V^T
  MatrixXd S = s.asDiagonal();
  MatrixXd approx = U * S * V.transpose();
  
  // 裁剪到 [0,1] 区间，防止出界
  for (int i = 0; i < approx.rows(); ++i) {
    for (int j = 0; j < approx.cols(); ++j) {
      if (approx(i, j) < 0.0) approx(i, j) = 0.0;
      if (approx(i, j) > 1.0) approx(i, j) = 1.0;
    }
  }
  
  // 转回 R 的 NumericMatrix
  Rcpp::NumericMatrix out(m, n);
  std::copy(approx.data(), approx.data() + m * n, out.begin());
  
  return out;
}

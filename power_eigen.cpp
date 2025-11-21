// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

// Power iteration using Eigen (dense)
// [[Rcpp::export]]
Rcpp::List power_iter_eigen_dense(const Rcpp::NumericMatrix &A, int n_iter = 1000) {
  int m = A.nrow();
  int n = A.ncol();
  
  // 映射 R 矩阵到 Eigen（列主序一致）
  Map<const MatrixXd> M(A.begin(), m, n);
  
  // 初始 v (n 维)
  VectorXd v = VectorXd::Random(n);
  v.normalize();
  
  for (int k = 0; k < n_iter; ++k) {
    VectorXd y = M * v;        // m 维
    VectorXd z = M.transpose() * y;   // n 维
    double nz = z.norm();
    if (nz == 0.0) break;
    v = z / nz;
  }
  
  VectorXd Av = M * v;
  double sigma1 = Av.norm();
  VectorXd u = Av / sigma1;
  
  Rcpp::NumericVector u_out(m), v_out(n);
  for (int i = 0; i < m; ++i) u_out[i] = u(i);
  for (int j = 0; j < n; ++j) v_out[j] = v(j);
  
  return Rcpp::List::create(
    Rcpp::Named("sigma1") = sigma1,
    Rcpp::Named("u1")     = u_out,
    Rcpp::Named("v1")     = v_out
  );
}

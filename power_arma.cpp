// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

// Power iteration for largest singular value and singular vectors
// A: m x n matrix (image), n_iter: iterations (e.g. 1000)

// [[Rcpp::export]]
Rcpp::List power_iter_arma(const Rcpp::NumericMatrix &A, int n_iter = 1000) {
  int m = A.nrow();
  int n = A.ncol();
  
  // 直接用 Armadillo 视图，不拷贝
  mat M( const_cast<double*>(A.begin()), m, n, false );
  
  // 初始随机右奇异向量 v (长度 n)
  vec v = randn<vec>(n);
  v /= norm(v, 2);
  
  for (int k = 0; k < n_iter; ++k) {
    // y = A * v  (m 维)
    vec y = M * v;
    // z = A^T * y = (A^T A) v  (n 维)
    vec z = M.t() * y;
    double nz = norm(z, 2);
    if (nz == 0.0) break;
    v = z / nz;
  }
  
  vec Av = M * v;
  double sigma1 = norm(Av, 2);
  vec u = Av / sigma1;
  
  return Rcpp::List::create(
    Rcpp::Named("sigma1") = sigma1,
    Rcpp::Named("u1")     = u,
    Rcpp::Named("v1")     = v
  );
}

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
double logpdf_mvn(const vec&x, const vec& mu, const mat& Sigma){
  const int d = x.n_elem;                 // 维度 d
  mat L = chol(Sigma, "lower");           // Cholesky:  Σ = L * L^T（要求 Σ 对称正定）
  vec z = solve(trimatl(L), x - mu);      // 解 L z = (x-μ)  —— 三角方程，避免显式 Σ^{-1}
  double quad = dot(z, z);                // (x-μ)^T Σ^{-1} (x-μ) = ||z||^2
  double logdet = 2.0 * sum(log(L.diag())); // log det Σ = 2 * ∑ log L_ii
  return -0.5*(d*log(2.0*M_PI) + logdet + quad); // 代回公式
}

// logf(x) = -1/2 (dlog(2pi) + logdetΣ + (x-μ)'Σ^{}-1}(x-μ)) 

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat add_eye(const arma::mat& A) {
  arma::mat B = A;
  B.diag() += 1.0;
  return B;
}

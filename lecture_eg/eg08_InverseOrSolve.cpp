#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
vec computeInverse(const mat& A, const vec& b, const vec& x_true){
  vec out(2);
  mat Ainv = inv(A);
  vec xhat = Ainv * b;
  out(0) = norm(xhat - x_true, 2);
  out(1) = norm(A*xhat - b,2);
  return out;
}

// [[Rcpp::export]]
vec SolveLS(const mat& A, const vec& b, const vec& x_true){
  vec out(2);
  vec xhat = solve(A,b);
  out(0) = norm(xhat - x_true,2);
  out(1) = norm(A*xhat - b,2);
  return out;
}
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// let's set: conv(x,y) = sum_i x[i]*y[n-i-1]

// [[Rcpp::export]]
double convolve_c(const arma::vec& x, const arma::vec& y){
  if(x.n_elem != y.n_elem) Rcpp::stop("x and y must have same length");
  const uword n = x.n_elem;
  double s = 0.0;
  for (uword i=0; i<n; ++i){
    s += x(i)*y(n-i-1);
  }
  return s;
}
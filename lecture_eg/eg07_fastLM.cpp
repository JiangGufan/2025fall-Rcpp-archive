#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
Rcpp::List fastLM(const vec& y, const mat& X){
  // beta\hat = (X'X)^{-1}X'y
  // sigma^2 = ||y-X\beta||^2/(n-p)
  const int n = X.n_rows, p = X.n_cols;
  mat XtX = X.t() * X;
  vec XtY = X.t() * y;
  vec beta = solve(XtX, XtY);
  vec resid = y-X*beta;
  double s2 = as_scalar((resid.t()*resid)/(n-p));
  vec se = sqrt(s2 *diagvec(XtX.i()));
  return Rcpp::List::create(
    Rcpp::Named("beta")=beta,
    Rcpp::Named("se")=se,
    Rcpp::Named("sigma")=sqrt(s2)
  );
}



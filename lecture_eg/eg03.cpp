#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
using namespace arma;

// [[Rcpp::export]]
arma::vec vectorSquared(const arma::vec& x){
  vec y(x.n_elem);
  for (uword i=0; i<x.n_elem; ++i){ // uword 是 Armadillo 自带的类型定义，等价于一个无符号整数unsigned integer，通常用于索引
    y(i)= std::pow(x(i),2.0);
  }
  return y;
}

// [[Rcpp::export]]
arma::mat matrixSquared(const arma::mat& X){
  mat Y(X.n_rows, X.n_cols);
  for (uword i=0; i<X.n_rows; ++i)
    for (uword j=0;j<X.n_cols;++j)
      Y(i,j) = std::pow(X(i,j),2.0);
  return Y;
}
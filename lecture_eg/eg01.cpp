#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
arma::vec timesTwo(arma::vec x){
  vec y;
  y = 2*x;
  return y;
}
// [[Rcpp::export]]
arma::mat matrixTimesTwo(arma::mat x){
  mat y;
  y = 2*x;
  return y;
}


// > Rcpp::sourceCpp("MY_RUC/Rcoding/Rcpp/eg01.cpp")
//   > xVector = 1:5
//   > timesTwo(xVector)
//   [,1]
// [1,]    2
// [2,]    4
// [3,]    6
// [4,]    8
// [5,]   10
// > yMatrix
// [,1] [,2] [,3]
// [1,]    1    3    5
// [2,]    2    4    6
// > matrixTimesTwo(yMatrix)
//   [,1] [,2] [,3]
// [1,]    2    6   10
// [2,]    4    8   12


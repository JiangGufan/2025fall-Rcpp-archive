#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// version1:三重循环
// [[Rcpp::export]]
arma::mat matrixMult_v1(const arma::mat& A,const arma::mat& B){
  const uword p=A.n_rows, r = A.n_cols, q = B.n_cols;
  // p*r ** r*q --> p*q
  mat C(p,q,fill::zeros);
  for (uword i=0;i<p;++i)
    for (uword j=0; j<q;++j){
      double s =0.0;
      for (uword k =0; k<r;++k)
        s +=A(i,k)*B(k,j);
      C(i,j) = s;
    }
  return C;
}


// version2:先转置B
// [[Rcpp::export]]
arma::mat matrixMult_v2(const arma::mat& A,const arma::mat& B){
  const uword p=A.n_rows, r = A.n_cols, q = B.n_cols;
  // p*r ** r*q --> p*q
  if (r!=B.n_rows) Rcpp::stop("ncol(A) must be nrow(B)");
  mat Bt = trans(B);
  mat C(p,q,fill::zeros);
  for (uword j=0; j<q;++j){
    const rowvec& bj =Bt.row(j);
    for(uword i=0;i<p;++i){
      C(i,j) = dot(A.row(i),bj);
    }
  }
  return C;
}


// version3:直接交给 Armadillo
// [[Rcpp::export]]
arma::mat matrixMult_v3(const arma::mat& A,const arma::mat& B){
  return A * B;
}


#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;


// 内置函数工具：统一打包误差 (||x-x_true||2, ||Ax-b||2)
static inline vec pack_err(const mat& A, const vec&x, const vec& b, const vec& x_true){
  vec out(2);
  out(0)=norm(x-x_true,2);
  out(1)=norm(A*x - b,2);
  return out;
}

// 0) baseline：直接 solve（库内部会自动选 LU/QR/Cholesky 等）
// [[Rcpp::export]]
arma::vec SolveLS(const mat& A, const vec& b, const vec& x_true){
  vec xhat = solve(A, b);                  // 通用求解器
  return pack_err(A, xhat, b, x_true);
}

// 1) LU（带行置换的 PA = LU；通用方阵，支持不定/非对称）
// [[Rcpp::export]]
vec LU(const mat& A, const vec& b, const vec& x_true){
  // solve Ax = LUx = b
  // where L lower triangular and U upper triangular matrix
  mat L,U,P;
  lu(L,U,P,A); // PA=LU
  // Armadillo的lu()使用带行置换的LU分解,置换矩阵P(Permutation Matrix),用于记录行交换顺序
  vec y = solve(trimatl(L), P*b); //Ly=Pb
  vec x = solve(trimatu(U), y);   //Ux=y
  return pack_err(A, x, b, x_true);
}

// 2) LDLᵀ（无主元、适用于“对称正定 SPD”；不适合不定矩阵） 
// [[Rcpp::export]]
vec LDL(const mat& A, const vec& b, const vec& x_true){
  if (A.n_rows != A.n_cols) Rcpp::stop("ncol(A) must be equal to nrow(A)");
  
  // 强制对称化
  mat S = 0.5 *(A+A.t());
  const uword n = S.n_rows;
  mat L(n,n,fill::zeros);
  vec D(n,fill::zeros);
  
  for (uword i=0; i<n; ++i){
    double sum_di = 0.0;
    for(uword k = 0; k<i;++k) sum_di += L(i,k) * L(i,k) * D(k);
    double di = S(i,i) - sum_di;
    if(std::abs(di) <= 1e-14){return vec({arma::datum::nan,arma::datum::nan});}
    D(i) = di;
    L(i,i) = 1.0;
    
    for (uword j=i+1; j<n; ++j){
      double s = 0.0;
      for (uword k=0; k<i; ++k) s += L(j,k)*L(i,k)*D(k);
      L(j,i) = (S(j,i)-s)/D(i);
    }
  }
  vec y = solve(trimatl(L),b);
  vec z = y/D;      //  D z = y（逐元素）
  vec x = solve(trimatu(L.t()),z);
  
  return pack_err(A, x, b, x_true);
}


// 3) Cholesky（SPD：A = Rᵀ R）  
// [[Rcpp::export]]
vec Cholesky(const mat& A, const vec& b, const vec& x_true){
  mat R;
  if(!chol(R,A))Rcpp::stop("A must be symmetric positive definite");
  vec y = solve(trimatl(R.t()),b);
  vec x = solve(trimatu(R),y);
  return pack_err(A,x,b,x_true);
}

  
// 4) SVD 最小二乘/广义逆（适合病态/秩亏）
// [[Rcpp::export]]
vec SVD(const mat& A, const vec& b, const vec& x_true){
  mat U,V;
  vec s;
  svd_econ(U,s,V,A);
  
  double smax = s.max();
  double tol = std::max(A.n_rows,A.n_cols)*std::numeric_limits<double>::epsilon()*smax;
  
  vec invs = s;
  for(uword i =0; i<s.n_elem; ++i) invs(i) = (s(i)>tol)? 1.0 / s(i) :0.0;
  vec x = V * (invs % (U.t()*b));
  
  return pack_err(A,x,b,x_true);
}




  
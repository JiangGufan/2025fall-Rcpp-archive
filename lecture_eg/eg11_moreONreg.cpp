#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// 1) 对称矩阵平方根
// [[Rcpp::export]]
Rcpp::List symSqrt(const int n =50){
  // // 构造对称正定矩阵 B = A^T A
  mat A = randu<mat>(n,n);
  mat B = A.t()*A;
  
  vec eigval; mat eigvec;
  eig_sym(eigval,eigvec,B);// eig_sym：对称矩阵的实特征分解，返回升序的特征值 eigval 与正交特征向量矩阵 eigvec
  
  // 法一：S1 = V diag(sqrt(λ)) V^T
  mat S1 = eigvec * diagmat(sqrt(eigval))*eigvec.t();
  
  // 法二：对 V 的列“逐列乘” λ^{1/4}，再 S2 = V V^T
  vec scale = pow(eigval,0.25);
  mat V2 = eigvec;
  V2.each_row() %= scale.t(); // %= 在Armadillo里是逐元素相乘并赋值
  mat S2 = V2 * V2.t();
  
  // 验证 S S^T ≈ B，Frobenius 范数
  double err1 = norm(S1*S1.t()-B,"fro")/norm(B,"fro");
  double err2 = norm(S2*S2.t()-B,"fro")/norm(B,"fro");
  
  return Rcpp::List::create(Rcpp::Named("relErr_method1")=err1,
                            Rcpp::Named("relErr_method2")=err2);
}

// 2) 子矩阵/视图与索引技巧
// [[Rcpp::export]]
Rcpp::List subview_index(){
  mat A = zeros<mat>(5,10);
  // 三种等价子块写法
  A.submat(0,1, 2,3)       = randu<mat>(3,3);
  A(span(0,2), span(1,3))  = randu<mat>(3,3);
  A(0,1, size(3,3))        = randu<mat>(3,3);
  
  // 列视图 + head/tail
  A.col(1)         = randu<vec>(5);
  A.col(2).head(3) += 123.0; // 只加前 3 个元素（讲义也有这一行）  
  
  // 元素级筛选/赋值
  mat X = randu<mat>(5,5);
  vec picked = X.elem(find(X > 0.5)); // 布尔筛选到向量
  uvec idx = {2, 3, 6, 8};  // 位置集合
  X.elem(idx) = ones<vec>(4);         // 指定位置赋值
  
  return Rcpp::List::create(
    Rcpp::Named("A") = A,
    Rcpp::Named("X") = X,
    Rcpp::Named("picked_gt_0p5") = picked
  );
}


// 3) Rcpp中随机数的生成
// [[Rcpp::export]]
mat rngCpp(const int N){
  mat X(N, 4);
  X.col(0) = vec(Rcpp::runif(N, -1,  1)); // U[-1,1]
  X.col(1) = vec(Rcpp::rnorm(N, 0, 10));  // N(0, 10^2)
  X.col(2) = vec(Rcpp::rt(N, 5));         // t_df=5
  X.col(3) = vec(Rcpp::rbeta(N, 1, 1));   // Beta(1,1)
  return X;
}


// 4）计算一串 lambda 的 ridge 解与 LOOCV 误差（基于 SVD）
// [[Rcpp::export]]
Rcpp::List ridge_loocv_svd(const arma::mat& X, const arma::vec& y,
                           const arma::vec& lambdas){
  // SVD: X = U D V^T
  mat U, V; vec D;
  svd_econ(U, D, V, X, "both");
  
  const int n = X.n_rows, p = X.n_cols, L = lambdas.n_elem;
  mat Beta(p, L, fill::zeros);
  vec loocv(L, fill::zeros);

  // 预先计算 U^T y 以及 U 的逐行平方（用于 diag(Hλ)）
  vec Uty = U.t() * y;
  mat Usq = square(U); // elementwise square

  for(int k=0;k<L;++k){
    double lam = lambdas[k];
    // shrinkage 向量： D ./ (D% D + lam)
    vec s = D / (square(D) + lam);

    // β_λ = V * diag(s) * U^T y
    Beta.col(k) = V * (s % Uty);
  
    // 拟合: yhat = U * diag(D .* s) * U^T y
    vec Ds = D % s;              // D .* s
    vec yhat = U * (Ds % Uty);   // 等价于 U * diag(Ds) * U^T y

    // diag(Hλ) = row_i sum_j U(i,j)^2 * (D_j^2 / (D_j^2 + λ))
    vec w = square(D) / (square(D) + lam);
    vec h = Usq * w;  // n×r 乘 r×1

    // LOOCV = mean( ((y - yhat)/(1 - h))^2 )
    vec res = (y - yhat) / (1.0 - h);
    loocv[k] = mean(square(res));
  }

  return Rcpp::List::create(
    Rcpp::Named("Beta")  = Beta,
    Rcpp::Named("loocv") = loocv
  );
}


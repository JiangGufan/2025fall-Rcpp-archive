// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::Triplet;

// [[Rcpp::export]]
Rcpp::List power_iter_eigen_sparse(const Rcpp::IntegerVector &i,
                                   const Rcpp::IntegerVector &j,
                                   const Rcpp::NumericVector &x,
                                   int m, int n,
                                   int n_iter = 1000) {
  int nnz = x.size();
  
  // Triplet MUST specify <double>
  std::vector< Triplet<double> > triplets;
  triplets.reserve(nnz);
  
  for (int k = 0; k < nnz; ++k) {
    triplets.emplace_back( Triplet<double>(i[k], j[k], x[k]) );
  }
  
  SparseMatrix<double> M(m, n);
  M.setFromTriplets(triplets.begin(), triplets.end());
  
  VectorXd v = VectorXd::Random(n);
  v.normalize();
  
  for (int t = 0; t < n_iter; ++t) {
    VectorXd y = M * v;
    VectorXd z = M.transpose() * y;
    double nz = z.norm();
    if (nz == 0.0) break;
    v = z / nz;
  }
  
  VectorXd Av = M * v;
  double sigma1 = Av.norm();
  VectorXd u = Av / sigma1;
  
  Rcpp::NumericVector u_out(m), v_out(n);
  for (int r = 0; r < m; ++r) u_out[r] = u(r);
  for (int c = 0; c < n; ++c) v_out[c] = v(c);
  
  return Rcpp::List::create(
    Rcpp::Named("sigma1") = sigma1,
    Rcpp::Named("u1")     = u_out,
    Rcpp::Named("v1")     = v_out
  );
}

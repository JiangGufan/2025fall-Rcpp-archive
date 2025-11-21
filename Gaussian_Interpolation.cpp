#include <RcppEigen.h>
#include <vector>
using namespace Eigen;
// [[Rcpp::depends(RcppEigen)]]

typedef Eigen::Triplet<double> T;
typedef SparseMatrix<double> SpMat;

// Create the second order difference matrix of size (D-2)-by-D
// 产生L矩阵。它给每列预留足够的空间，然后直接安置元素。
SpMat createLmat(int D){
  SpMat Lmat(D-2, D);
  // VectorXd::Constant(D, 3) is a vector of size D, and = (3,3,...,3)^T
  Lmat.reserve(VectorXd::Constant(D, 3));
  int i, j, k;
  for(j = 0; j < D; j++){
    for(i = -1; i <  2; i++){
      k = i + j - 1;
      if(k < 0 || k > D-3) continue;
      if(i==0)
        Lmat.insert(k, j) = 1;
      else
        Lmat.insert(k, j) = -0.5;
    } // for i
  }// for j
  return Lmat;
}

// Create the observation matrix A.
// 产生A矩阵。它使用了生成稀疏矩阵的第二种方法，从三元组（行、列、值）生成。
SpMat createAmat(VectorXd jSet, int D){
  SpMat Amat(jSet.size(), D);
  int i, j, k;
  std::vector<T> tripletList;
  tripletList.reserve(jSet.size() );
  for(j = 0; j < jSet.size(); j++){
    tripletList.push_back(T(j, jSet(j), 1));
  }// for j
  
  Amat.setFromTriplets(tripletList.begin(), tripletList.end() );
  return Amat;
}


// 计算条件均值、条件方差。
// [[Rcpp::export]]
MatrixXd gpFitting(VectorXd jSet, VectorXd yVec, 
                   double lambdaSq, double sigmaSq_y, int D) {
  // The conditional mean, and its standard deviation uncertainty
  VectorXd mu, muSigma;
  
  SpMat Lmat, Amat, SigmaXYInverse; 
  Lmat = createLmat(D);
  Amat = createAmat(jSet, D);
  
  SigmaXYInverse = lambdaSq * Lmat.transpose() * Lmat 
  + Amat.transpose() * Amat / sigmaSq_y;
  
  //sparse solve to get the conditional mean
  Eigen::SimplicialLLT< SpMat > solver(SigmaXYInverse);
  // take inverse to get uncertinaty.
  mu = Amat.transpose() * yVec / sigmaSq_y;
  mu = solver.solve(mu);
  
  // compute the conditional variance
  MatrixXd Imat(D, D);
  Imat.setIdentity();
  auto SigmaSparse = solver.solve(Imat); // to get the inverse
  //MatrixXd SigmaDense = MatrixXd(SigmaSparse);
  //SigmaSparse.di
  muSigma = SigmaSparse.diagonal();
  
  MatrixXd result(D, 2);
  result << mu , muSigma.cwiseSqrt();
  return result;
}
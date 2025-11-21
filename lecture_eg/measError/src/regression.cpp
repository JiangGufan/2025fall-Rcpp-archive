#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

//' Compute the ordinary least squares.
//' 
//' @param D the the response vector
//' @param W the the predictor vector
//' @return the estamated regression coefficients
// [[Rcpp::export]]
arma::vec naiveRegression(arma::vec D, arma::vec W){
    vec betaHat(2);
    double Dbar, Wbar, d1, d2;
    Dbar = mean(D);
    Wbar = mean(W);
    d1 = sum((W - Wbar) % (D - Dbar));
    d2 = sum((W - Wbar) % (W - Wbar));
    betaHat(1) = d1 / d2;
    betaHat(0) = Dbar - Wbar * betaHat(1);
    return betaHat;
}

//' Compute the moment corrected estimator for the simple linear regression.
//' 
//' @param D the the response vector
//' @param W the the predictor vector
//' @param sigmaU the measurement uncertainty vector for the predictor
//' @return the estamated regression coefficients
//' @examples
//' library(measError)
//' set.seed(100)
//' n = 400
//' beta0 = 1
//' beta1 = 2
//' sigmaU = runif(n, 0.4, 0.6)
//' sigmaQ = runif(n, 0.1, 0.2)
//' sigmaE = 0.2
//' simuData = getData(n, beta0, beta1, sigmaU, sigmaQ, sigmaE)
//' (betaHat1 = naiveRegression(simuData$D, simuData$W))
//' (betaHat2 = correctedRegression(simuData$D, simuData$W, sigmaU))
// [[Rcpp::export]]
arma::vec correctedRegression(arma::vec D, arma::vec W, arma::vec sigmaU){
    vec betaHat(2);
    double Dbar, Wbar, sigmaU2bar, d1, d2;
    int N = D.n_elem;
    Dbar = mean(D);
    Wbar = mean(W);
    sigmaU2bar = mean(sigmaU % sigmaU);
    d1 = sum((W - Wbar) % (D - Dbar)) / (N-1);
    d2 = sum((W - Wbar) % (W - Wbar)) / (N-1);
    betaHat(1) = d1 / (d2 - sigmaU2bar);
    betaHat(0) = Dbar - Wbar * betaHat(1);
    return betaHat;
    
}

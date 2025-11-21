#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// mirror index
inline int mirrorIndex(int idx, int len){
  if (len<=1) return 0;
  int period = 2* (len-1);
  int t = idx % period;
  if (t <0) t += period;
  return (t <= len -1) ? t : (period-t);
}


// [[Rcpp::export]]
arma::vec localSmoothing (const arma::vec& y, const arma::vec& w){
  int n = (int) y.n_elem, k = (int) w.n_elem;
  if (k % 2 ==0) Rcpp::stop("w length must be odd");
  int half = k/2;
  vec out(n, fill::zeros);
  for (int i=0; i<n; ++i){
    double s =0.0;
    for (int j=-half; j<=half; ++j){
      int fetchI = mirrorIndex(i+j,n);
      s += w(j + half) * y(fetchI);
    }
    out(i) = s;
  }
  return out;
}
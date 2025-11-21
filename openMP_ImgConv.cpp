// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace arma;

// 镜像边界：idx ∈ Z, size = m (或 n)，返回 [0, size-1] 内的索引
inline int mirrorIndex(int idx, int size) {
  if (idx < 0)         return -idx;                 // -1 -> 1, -2 -> 2, ...
  if (idx >= size)     return 2*size - 2 - idx;     // size -> size-2, ...
  return idx;
}

// 串行版本：做 3×3 卷积
// [[Rcpp::export]]
mat imageConv_serial(const mat &img, const mat &kernel) {
  int m = img.n_rows;
  int n = img.n_cols;
  
  mat result(m, n, fill::zeros);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      double fin = 0.0;
    
      for (int h = -1; h <= 1; ++h) {
        for (int k = -1; k <= 1; ++k) {
          int imgI = mirrorIndex(i + h, m);
          int imgJ = mirrorIndex(j + k, n);
          fin += img(imgI, imgJ) * kernel(h + 1, k + 1);
        }
      }
      result(i, j) = fin;
    }
  }
  return result;
}

// OpenMP 并行版本：在 j 维度上并行
// nthreads 可指定线程数；=0 表示用默认线程数
// [[Rcpp::export]]
mat imageConv_parallel(const mat &img, const mat &kernel, int nthreads = 0) {
  int m = img.n_rows;
  int n = img.n_cols;
  mat result(m, n, fill::zeros);

#ifdef _OPENMP
  if (nthreads > 0)
    omp_set_num_threads(nthreads);

#pragma omp parallel for schedule(static) \
  default(none) shared(img, kernel, result, m, n)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        double fin = 0.0;
        
        for (int h = -1; h <= 1; ++h) {
          for (int k = -1; k <= 1; ++k) {
            int imgI = mirrorIndex(i + h, m);
            int imgJ = mirrorIndex(j + k, n);
            fin += img(imgI, imgJ) * kernel(h + 1, k + 1);
          }
        }
        result(i, j) = fin;
      }
    }
#else
    // 没有 OpenMP 时退化成串行
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        double fin = 0.0;
        
        for (int h = -1; h <= 1; ++h) {
          for (int k = -1; k <= 1; ++k) {
            int imgI = mirrorIndex(i + h, m);
            int imgJ = mirrorIndex(j + k, n);
            fin += img(imgI, imgJ) * kernel(h + 1, k + 1);
          }
        }
        result(i, j) = fin;
      }
    }
#endif
    
    return result;
}

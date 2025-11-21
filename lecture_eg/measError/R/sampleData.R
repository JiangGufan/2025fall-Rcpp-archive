#' Get simulation data
#' 
#' @param n the sample size
#' @param beta0 the model intercept
#' @param beta1 the model slope
#' @param sigmaU the measurement uncerntainty for X
#' @param sigmaQ the measurement uncertainty for Y
#' @param sigmaE the model error
#' @return the generated data as a data frame
#' @export
getData = function(n, beta0, beta1, sigmaU, sigmaQ, sigmaE){
    X = runif(n,  -2, 2)
    Y = beta0 + beta1 * X + rnorm(n, sd = sigmaE)
    D = Y + rnorm(n, sd = sigmaQ)
    W = X + rnorm(n, sd = sigmaU)
    return(data.frame(D = D, W = W))
}



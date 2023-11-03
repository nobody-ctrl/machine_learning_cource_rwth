import numpy as np

def computeGaussian(X, mean, covariance):
    D = len(mean)
    #if np.linalg.det(covariance) == 0:
      #print("Matrix is singular")
      #return 0
    result = 1/((np.sqrt(2*np.pi))**D*np.sqrt(np.linalg.det(covariance)))*np.exp(-1/2*np.matmul(np.matmul((X-mean).transpose(), np.linalg.inv(covariance)), (X-mean)))
    return result

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    #first get the data 
    (N, D) = X.shape
    K = len(means)
    #start with outloop
    logLikelihood = 0
    for n in range(0, N):
      result_inner_loop = 0
      #compute the ln()...
      for k in range(0, K):
        result_inner_loop += weights[k]*computeGaussian(X[n], means[k], covariances[:, :, k])
      logLikelihood += np.log(result_inner_loop)
    return logLikelihood


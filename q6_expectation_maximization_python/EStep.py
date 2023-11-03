import numpy as np
from getLogLikelihood import getLogLikelihood
from getLogLikelihood import computeGaussian

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    K = len(means)
    (N, D) = X.shape
    gamma = np.empty([N, K])
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    for n in range(0, N):
      for k in range(0, K):
        #compute sum in the denomenator
        sum = 0
        for j in range(0, K):
          sum += weights[j]*computeGaussian(X[n], means[j], covariances[:, :, j])
        gamma[n][k] = weights[k]*computeGaussian(X[n], means[k], covariances[:, :, k])/sum
    return [logLikelihood, gamma]

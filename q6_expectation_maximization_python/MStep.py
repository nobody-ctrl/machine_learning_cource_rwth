import numpy as np
from getLogLikelihood import getLogLikelihood
from getLogLikelihood import computeGaussian

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    (N, K) = gamma.shape
    (N, D) = X.shape

    #print("DATA WE HAVE..............................")
    #print("N=", N, ", K=", K, ", D=", D)
    #print("X=", X)
    #print("Gamma=", gamma)
    #print("END END END DATA WE HAVE..............................")
    #Compute N_k
    Ni = np.empty(K)
    weights = np.empty(K)
    means = np.empty([K, D])
    covariances = np.empty([D, D, K])
    for k in range(0, K):
      Ni[k] = 0
      for n in range(0, N):
        Ni[k] += gamma[n][k]
    print("Vector Ni K-dim looks like ", Ni)
    #Now compute weights, means and covariances
    for k in range(0, K):
      weights[k] = Ni[k]/N
      print("[+] Doing ", k, "-th iteration...", weights[k])
      #compute sum first
      sum = 0
      for y in range(0, N):
        sum += gamma[y][k]*X[y]
      means[k] = 1/Ni[k]*sum
      #compute sum first
      #sumx = np.empty([D, D])
      #for y in range(0, N):
        #sumx += gamma[y][k]*np.matmul((X[y]-means[k]), (X[y]-means[k]).transpose())
      inner_cov = np.sum([gamma[n, k] * np.tensordot((X[n] - means[k]), (X[n] - means[k]), axes=0) for n in range(N)], axis=0)
      covariances[:, :, k] = 1/Ni[k]*np.asarray(inner_cov)
      #print("covariances[:, :, k]=", covariances[:, :, k])
    #print("XXXXXXXXXXXXXX")
    #for k in range(0, K):
      #print(covariances[:, :, k])
    #print("XXXXXXXXXXXXXX")
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood

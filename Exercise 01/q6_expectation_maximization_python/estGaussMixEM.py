import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    #####Insert your code here for subtask 6e#####
    (n_dim, D) = data.shape
    #Firstly initialize
    weights = np.ones(K)/K
    covariances = np.empty([D, D, K])
    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_
    # Create initial covariance matrices
    for j in range(K):
      data_cluster = data[cluster_idx == j]
      min_dist = np.inf
      for i in range(K):
        # compute sum of distances in cluster
        dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
        if dist < min_dist:
          min_dist = dist
      covariances[:, :, j] = np.eye(D) * min_dist
    #Do regularization to make sure that matrix is invertible
    for k in range(0, K):
      covariances[:, :, k] = regularize_cov(covariances[:, :, k], epsilon)
    #Then perform E-Step of EM Algorithm
    [logLikelihood, gamma] = EStep(means, covariances, weights, data)
    #Then perform M-Step of EM Algorithm
    [weights, means, covariances, logLikelihood] = MStep(gamma, data)
    return [weights, means, covariances]

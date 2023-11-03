import numpy as np

def get_pr(x, k, samples, N):
  radius = 0
  neighbors = 0
  while neighbors < k:
    radius += 0.08
    bool_arr = (samples-x)**2 < radius**2
    neighbors = np.sum(bool_arr) 
    volume = np.pi*(radius**2)
    probab = k/(N*volume)
  return probab

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    pos = np.arange(-5,5,0.1)

    if len(samples.shape) == 1:
        N = samples.shape[0]
        D = 1
    else:
        D = samples.shape[0]
        N = samples.shape[1]
    
    est_probabilities = []
    for i in pos:
        est_probabilities.append(get_pr(i,k, samples, N))
    estDensity = np.column_stack((pos, est_probabilities))
    return estDensity
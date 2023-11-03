import numpy as np
from scipy.integrate import quad

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    pos = np.arange(-5,5,0.1)

    if len(samples.shape) == 1:
        N = samples.shape[0]
        D = 1
    else:
        D = samples.shape[0]
        N = samples.shape[1]
    
    ## Define gaussian function for gaussian kernel
    def gaussian(x_vec, h):
        return np.sqrt(1/((2*np.pi**h)))*np.exp(-np.inner(x_vec, x_vec)/(2*h**2))
    
    ## check for pdf to be normalized
    volume = quad(gaussian,np.min(samples),np.max(samples), args=h)[0]
    
    ## estimate value of probability density at each data point
    estimated_probabilities = []
    
    for i,j in enumerate(pos):
        estimated_probab = 1/(N*volume) * (np.sum(np.asarray([gaussian(j-k,h) for k in samples])))
        estimated_probabilities.append(estimated_probab)
    
    ## Give estDensity the desired shape
    estDensity = np.column_stack((pos, estimated_probabilities))

    # print("estDensity for kde:", estDensity)
    return estDensity

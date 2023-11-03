import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood
from PIL import Image
from getLogLikelihood import computeGaussian

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel
    print(img.shape)
    height, width, par = img.shape
    #####Insert your code here for subtask 1g#####
    #result = np.random.random((height, width))
    result = np.zeros((height, width, 3), dtype=np.uint8)
    #try firstly classify skin
    (weights, means, covariances) = estGaussMixEM(sdata, K, n_iter, epsilon)
    (weights_a, means_a, covariances_a) = estGaussMixEM(ndata, K, n_iter, epsilon)
    for i in range(0, height):
      for j in range(0, width):
        p_skin = 0
        for k in range(0, K):
          p_skin += weights[k]*computeGaussian(img[i][j], means[k], covariances[:, :, k])
        p_nskin= 0
        for k in range(0, K):
          p_nskin += weights_a[k]*computeGaussian(img[i][j], means_a[k], covariances_a[:, :, k])
        if (p_skin > theta*p_nskin):
          result[i][j] = (255,255,255)
        else:
          result[i][j] = (0, 0, 0)
    print("Weights=", weights)
    print("Means=", means)
    print("Covariances=", covariances)
    return result

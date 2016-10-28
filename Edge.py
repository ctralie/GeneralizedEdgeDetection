import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import sys
sys.path.append('GeometricCoverSongs')
from CurvatureTools import getCurvVectors

#I: M x N grayscale image
#i: Row which is at the top of each patch
#TODO: Weight by gaussian centered at patch?
def getPatchRow(I, i, dim):
    [M, N] = [I.shape[0], I.shape[1]]
    NCols = N - dim + 1
    ret = np.zeros((NCols, dim*dim))
    PatchRow = I[i:i+dim, :]
    for j in range(NCols):
        ret[j, :] = PatchRow[:, j:j+dim].flatten()
    return ret

def getHorizontalJumps(I, dim, sigma = 1):
    [M, N] = [I.shape[0], I.shape[1]]
    NRows = M - dim + 1
    NCols = N - dim + 1
    Jumps = np.zeros((NRows, NCols))
    for i in range(NRows):
        print "%i of %i"%(i, NRows)
        X = getPatchRow(I, i, dim)
        [XSmooth, Vel, Curv, Tors] = getCurvVectors(X, 3, sigma)
        Jumps[i, :] = np.sqrt(np.sum(Tors**2, 1))
    return Jumps

def getVerticalJumps(I, dim, sigma = 1):
    return getHorizontalJumps(I.T, dim, sigma).T

if __name__ == "__main__":
    I = color.rgb2gray(io.imread('mandrill.jpg'));
    dim = 10
    sigma = 2
    JumpsH = getHorizontalJumps(I, dim, sigma)
    JumpsV = getVerticalJumps(I, dim, sigma)
    Jumps = np.sqrt(JumpsH**2 + JumpsV**2)
    plt.subplot(221)
    plt.title("Tors dim %i sigma %g"%(dim, sigma))
    plt.imshow(I, cmap='Greys_r')
    plt.subplot(222)
    plt.imshow(Jumps)
    plt.subplot(223)
    plt.imshow(JumpsH)
    plt.title('Horizontal Jumps')
    plt.subplot(224)
    plt.imshow(JumpsV)
    plt.title('Vertical Jumps')

    plt.show()

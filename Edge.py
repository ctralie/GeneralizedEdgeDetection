import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import sys
sys.path.append('GeometricCoverSongs')
from CurvatureTools import getCurvVectors

#I: M x N x Ch image
#i: Row which is at the top of each patch
#TODO: Weight by gaussian centered at patch?
def getPatchRow(I, i, dim):
    assert I.ndim == 2 or I.ndim == 3, "Input image array should be of dimension 2 or 3!"
    if I.ndim == 2:
        I = I.reshape((I.shape[0], I.shape[1], 1))

    M, N, Ch = I.shape
    NCols = N - dim + 1
    ret = np.zeros((NCols, dim*dim*Ch))
    PatchRow = I[i:i+dim, :,:]
    for j in range(NCols):
        ret[j, :] = PatchRow[:, j:j+dim,:].flatten()
    return ret

def getHorizontalStats(I, dim, sigma = 1):
    assert I.ndim == 2 or I.ndim == 3, "Input image array should be of dimension 2 or 3!"
    if I.ndim == 2:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    M, N, Ch = I.shape
    NRows = M - dim + 1
    NCols = N - dim + 1
    Jumps = np.zeros((NRows, NCols))
    Curvs = np.zeros((NRows, NCols))
    Tors = np.zeros((NRows, NCols))
    for i in range(NRows):
        print "%i of %i"%(i, NRows)
        X = getPatchRow(I, i, dim)
        [XSmooth, Vel, Curv, Tor] = getCurvVectors(X, 3, sigma)
        Jumps[i, :] = np.sqrt(np.sum(Vel**2, 1))
        Curvs[i, :] = np.sqrt(np.sum(Curv**2, 1))
        Tors[i, :] = np.sqrt(np.sum(Tor**2, 1))
    return (Jumps, Curvs, Tors)

def getVerticalStats(I, dim, sigma = 1):
    (J, C, T) = getHorizontalStats(np.swapaxes(I,0,1), dim, sigma)
    return (J.T, C.T, T.T)

def getStats(I, dim, sigma = 1):
    (JumpsH, CurvsH, TorsH) = getHorizontalStats(I, dim, sigma)
    (JumpsV, CurvsV, TorsV) = getVerticalStats(I, dim, sigma)
    Jumps = np.sqrt(JumpsH ** 2 + JumpsV ** 2)
    Curvs = np.sqrt(CurvsH ** 2 + CurvsV ** 2)
    Tors = np.sqrt(TorsH ** 2 + TorsV ** 2)
    return Jumps, Curvs, Tors, JumpsH, CurvsH, TorsH, JumpsV, CurvsV, TorsV

if __name__ == "__main__":
    filename = "mandrill.jpg"

    dim = 5
    sigma = 2

    I = color.rgb2gray(io.imread(filename));
    Jumps, Curvs, Tors, JumpsH, CurvsH, TorsH, JumpsV, CurvsV, TorsV = getStats(I, dim, sigma = sigma)

    plt.subplot(431)
    plt.title("dim %i sigma %g"%(dim, sigma))
    plt.imshow(I, cmap='Greys_r')
    plt.axis('off')

    plt.subplot(434)
    plt.imshow(JumpsH)
    plt.axis('off')
    plt.title('Horizontal  Jumps')
    plt.subplot(435)
    plt.imshow(JumpsV)
    plt.axis('off')
    plt.title('Vertical Jumps')
    plt.subplot(436)
    plt.imshow(Jumps)
    plt.axis('off')
    plt.title('Jumps')


    plt.subplot(437)
    plt.imshow(CurvsH)
    plt.axis('off')
    plt.title('Horizontal  Curvs')
    plt.subplot(438)
    plt.imshow(CurvsV)
    plt.axis('off')
    plt.title('Vertical Curvs')
    plt.subplot(439)
    plt.imshow(Curvs)
    plt.axis('off')
    plt.title('Curvs')

    plt.subplot(4, 3, 10)
    plt.imshow(TorsH)
    plt.axis('off')
    plt.title('Horizontal  Tors')
    plt.subplot(4, 3, 11)
    plt.imshow(TorsV)
    plt.axis('off')
    plt.title('Vertical Tors')
    plt.subplot(4, 3, 12)
    plt.imshow(Tors)
    plt.axis('off')
    plt.title('Tors')

    plt.savefig("Results/Results.png", bbox_inches = "tight", dpi = 300)

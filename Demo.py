from VGG19network import make_VGG16_function
from Edge import getStats
import matplotlib.pyplot as plt

def plot_results(savefile, Iorig, Jumps, Curvs, Tors, JumpsH, CurvsH, TorsH, JumpsV, CurvsV, TorsV):
    plt.subplot(431)
    plt.title("dim %i sigma %g" % (dim, sigma))
    plt.imshow(Iorig/Iorig.max())
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

    plt.savefig(savefile, bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    imgname = "mandrill"
    savename = "Results/" + imgname
    from skimage import io
    import numpy as np

    dim = 5
    sigma = 2

    # initialize conv net
    theano_VGG19 = make_VGG16_function()

    # list of layers to extract features from
    # 'input' corresponds to the original RGB image
    layer_list = ['input', 'pool1', 'pool2', 'pool3']

    features, Iorig = theano_VGG19(imgname + '.jpg', layers=layer_list)
    for l in layer_list:
        Jumps, Curvs, Tors, JumpsH, CurvsH, TorsH, JumpsV, CurvsV, TorsV = getStats(features[l], dim, sigma=sigma)
        plot_results(savename + '_' + l + '.png', Iorig, Jumps, Curvs, Tors, JumpsH, CurvsH, TorsH, JumpsV, CurvsV, TorsV)





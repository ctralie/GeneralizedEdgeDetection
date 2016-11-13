# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

import cPickle
import numpy as np
#import theano
#import theano.tensor as T

from lasagne.layers import get_output
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax, rectify


# change this variable to accept different image sizes
#IM_SIZE = (3,224,224)
IM_SIZE = (3,512,512)

VGG19_PARAM_LIST = ['conv1_1','conv1_2',
                    'conv2_1', 'conv2_2',
                    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

def VGG19_net(input_shape=(None,) + IM_SIZE, pad =  (1,1), only_conv = False):
    """ Defines VGG19 network in Lasagne """
    net = {}

    net['input'] = InputLayer(input_shape)
    net['conv1_1'] = ConvLayer(net['input'], num_filters=64, filter_size=(3,3), pad=pad, flip_filters=False, nonlinearity = None)
    net['relu1_1'] = NonlinearityLayer(net['conv1_1'], nonlinearity=rectify)
    net['conv1_2'] = ConvLayer(net['relu1_1'], num_filters=64, filter_size=(3,3), pad=pad, flip_filters=False, nonlinearity = None)
    net['relu1_2'] = NonlinearityLayer(net['conv1_2'], nonlinearity=rectify)
    net['pool1'] = PoolLayer(net['relu1_2'], pool_size=2, stride = 2, mode='max',ignore_border=False)
    net['conv2_1'] = ConvLayer(net['pool1'], num_filters=128, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu2_1'] = NonlinearityLayer(net['conv2_1'], nonlinearity=rectify)
    net['conv2_2'] = ConvLayer(net['relu2_1'], num_filters=128, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu2_2'] = NonlinearityLayer(net['conv2_2'], nonlinearity=rectify)
    net['pool2'] = PoolLayer(net['relu2_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv3_1'] = ConvLayer(net['pool2'], num_filters=256, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu3_1'] = NonlinearityLayer(net['conv3_1'], nonlinearity=rectify)
    net['conv3_2'] = ConvLayer(net['relu3_1'], num_filters=256, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu3_2'] = NonlinearityLayer(net['conv3_2'], nonlinearity=rectify)
    net['conv3_3'] = ConvLayer(net['relu3_2'], num_filters=256, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu3_3'] = NonlinearityLayer(net['conv3_3'], nonlinearity=rectify)
    net['conv3_4'] = ConvLayer(net['relu3_3'], num_filters=256, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu3_4'] = NonlinearityLayer(net['conv3_4'], nonlinearity=rectify)
    net['pool3'] = PoolLayer(net['relu3_4'], pool_size=2, stride=2, mode='max', ignore_border=False)

    net['conv4_1'] = ConvLayer(net['pool3'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu4_1'] = NonlinearityLayer(net['conv4_1'], nonlinearity=rectify)
    net['conv4_2'] = ConvLayer(net['relu4_1'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False, nonlinearity=None)
    net['relu4_2'] = NonlinearityLayer(net['conv4_2'], nonlinearity=rectify)
    net['conv4_3'] = ConvLayer(net['relu4_2'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu4_3'] = NonlinearityLayer(net['conv4_3'], nonlinearity=rectify)
    net['conv4_4'] = ConvLayer(net['relu4_3'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu4_4'] = NonlinearityLayer(net['conv4_4'], nonlinearity=rectify)
    net['pool4'] = PoolLayer(net['relu4_4'], pool_size=2, stride=2, mode='max', ignore_border=False)

    net['conv5_1'] = ConvLayer(net['pool4'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False, nonlinearity=None)
    net['relu5_1'] = NonlinearityLayer(net['conv5_1'], nonlinearity=rectify)
    net['conv5_2'] = ConvLayer(net['relu5_1'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu5_2'] = NonlinearityLayer(net['conv5_2'], nonlinearity=rectify)
    net['conv5_3'] = ConvLayer(net['relu5_2'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu5_3'] = NonlinearityLayer(net['conv5_3'], nonlinearity=rectify)
    net['conv5_4'] = ConvLayer(net['relu5_3'], num_filters=512, filter_size=(3, 3), pad=pad, flip_filters=False,nonlinearity=None)
    net['relu5_4'] = NonlinearityLayer(net['conv5_4'], nonlinearity=rectify)
    net['pool5'] = PoolLayer(net['relu5_4'], pool_size=2, stride=2, mode='max', ignore_border=False)

    if not only_conv:
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096, nonlinearity=None)
        net['relu6'] = NonlinearityLayer(net['fc6'], nonlinearity=rectify)
        net['fc7'] = DenseLayer(net['relu6'], num_units=4096, nonlinearity=None)
        net['relu7'] = NonlinearityLayer(net['fc7'], nonlinearity=rectify)
        net['fc8'] = DenseLayer(net['relu7'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], nonlinearity=softmax)
    return net

def load_VGG19(only_conv=True):
    """Initializes VGG19 networks with the pre-trained weights.
    We only use the convolutional layers, which allows us to consider images of different size."""

    net = VGG19_net(only_conv=only_conv)
    file = 'vgg19.pkl'
    with open(file) as f:
        params = cPickle.load(f)

    input_mean = params['mean value']
    params = params['param values']
    for i, p in enumerate(params):
        if i/2 >= len(VGG19_PARAM_LIST):
            break
        layer_name = VGG19_PARAM_LIST[i/2]
        if p.ndim == 1: # bias
            net[layer_name].b.set_value(p)
        elif p.ndim == 4: # conv kernel
            # do we need to flip layer, I think not
            net[layer_name].W.set_value(p)
        elif p.ndim == 2:  # dense layer
            net[layer_name].W.set_value(p)
        #print('Layer %s is set.' % (layer_name))
    return net, input_mean


def make_VGG16_function():
    """ Create two functions: a functions that loads jpg image and preprocesses it for VGG19;
                              a theano function that extracts features from VGG19 network

        theano_VGG19 return feature map of format (height x width x number of channels) for a given list of layers
    """

    net, input_mean = load_VGG19()
    def load_image(img_file):
        from skimage.io import imread
        from skimage.transform import resize

        # load image from file and resize
        I = resize(imread(img_file), (IM_SIZE[1], IM_SIZE[2]), preserve_range=True)
        # if image is grayscale, add channels by copying the gray channel
        if I.ndim == 2:
            I_ = np.copy(I)
            I = np.zeros((IM_SIZE[1], IM_SIZE[2], 3))
            for i in range(0, 3):
                I[:, :, i] = I_
        Iorig = np.copy(I)

        # shuffle axes to match the expected VGG19 format
        I = np.swapaxes(np.swapaxes(I, 1, 2), 0, 1)
        # convert from RGB to BGR
        I = I[::-1, :, :]
        I = I - input_mean[:, np.newaxis, np.newaxis]
        # return a 4D tensor expecte by VGG19
        return I[np.newaxis], Iorig

    def theano_VGG19(img_file, layers = ['pool1']):
        """ Get output of the layers in layers """
        I, Iorig = load_image(img_file)
        features = {}
        for l in layers:
            print('Extracting features of layer %s' % (l))
            # evaluate the features and change the format back to height x width x channels
            f_ = get_output(net[l],inputs=I).eval()[0]
            #print f_.shape
            features[l] = np.swapaxes(np.swapaxes(f_,0,1),1,2)
        return features, Iorig

    return theano_VGG19

if __name__ == '__main__':
    """ Example of how to extract features from VGG19 network"""

    theano_VGG19 = make_VGG16_function()
    features, Iorig = theano_VGG19('mandrill.jpg', layers=['pool1', 'pool2'])
    for name, f in features.items():
        print name
        print f.shape




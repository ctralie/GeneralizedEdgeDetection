# Generalized Edge Detection

##  Setup

Install [theano](https://github.com/Theano/Theano) and [lasagne](http://lasagne.readthedocs.io/en/latest/).

```
$ pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
$ pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

Clone the repository.
```
$ git clone https://github.com/jureso/GeneralizedEdgeDetection.git
$ cd GeneralizedEdgeDetection
```

Get the submodules.
```
$ git submodule init
$ git submodule update
```

Download VGG19 weights.
```
$ wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
```

Run demo.
```
$ python Demo.py
```
See results in the **Results** folder

## Examples

See **Demo.py**
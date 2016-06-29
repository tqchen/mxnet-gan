"""Collection of encoder symbols

An encoder encode a input to a vector space.
"""
import mxnet as mx
from .ops import conv2d_bn_leaky

def lenet(data=None):
    """Lenet before classification layer."""
    data = mx.sym.Variable("data") if data is None else data
    # 28x28
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20, name="conv1")
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50, name="conv2")
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    d5 = mx.sym.Flatten(pool2)
    d5 = mx.sym.FullyConnected(d5, num_hidden=500, name="fc1")
    d5 = mx.sym.Activation(d5, act_type="tanh")
    return d5


def dcgan(data=None, ngf=128):
    """Conv net used in original DGCAN"""
    data = mx.sym.Variable("data") if data is None else data
    # 128, 16, 16
    net = mx.sym.Convolution(data, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf, name="e1_conv")
    net = mx.sym.LeakyReLU(net, slope=0.2, act_type="leaky", name="e1_act")
    # 256, 8, 8
    net = conv2d_bn_leaky(net, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf*2, prefix="e2")
    # 512, 4, 4
    net = conv2d_bn_leaky(net, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf*4, prefix="e3")
    net = mx.sym.Flatten(net)
    return net

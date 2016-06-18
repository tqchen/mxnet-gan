"""Collection of operatpors."""
import numpy as np
import mxnet as mx
from .log_sum_exp import log_sum_exp

BatchNorm = mx.sym.CuDNNBatchNorm
eps = 1e-5 + 1e-12

def deconv2d(data, ishape, oshape, kshape, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    # osize = stride * (isize - 1) + ksize - 2 * pad
    # pad = (stride * (isize - 1) + ksize - osize) / 2
    pad0 = (stride[0] * (ishape[-2] - 1) + kshape[0] - oshape[-2])
    pad1 = (stride[1] * (ishape[-1] - 1) + kshape[1] - oshape[-1])
    assert pad0 >= 0
    assert pad1 >= 0
    assert pad0 % 2 == 0
    assert pad1 % 2 == 0
    net = mx.sym.Deconvolution(data,
                               kernel=kshape,
                               stride=stride,
                               pad=(pad0 / 2, pad1/2),
                               num_filter=oshape[0],
                               no_bias=True,
                               name=name)
    return net


def deconv2d_bn_relu(data, prefix , **kwargs):
    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    net = BatchNorm(net, fix_gamma=True, eps=eps, name="%s_bn" % prefix)
    net = mx.sym.Activation(net, name="%s_act" % prefix, act_type='relu')
    return net


def deconv2d_act(data, prefix, act_type="relu", **kwargs):
    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    net = mx.sym.Activation(net, name="%s_act" % prefix, act_type=act_type)
    return net


def conv2d_bn_leaky(data, prefix, **kwargs):
    net = mx.sym.Convolution(data, name="%s_conv" % prefix, **kwargs)
    net = BatchNorm(net, fix_gamma=True, eps=eps, name="%s_bn" % prefix)
    net = mx.sym.LeakyReLU(net, act_type="leaky", name="%s_leaky" % prefix)
    return net

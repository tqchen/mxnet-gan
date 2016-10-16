"""Collection of operatpors."""
import numpy as np
import mxnet as mx
from .custom_ops import log_sum_exp
from .custom_ops import constant

BatchNorm = mx.sym.BatchNorm
eps = 1e-5 + 1e-12

def deconv2d(data, ishape, oshape, kshape, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    target_shape = (oshape[-2], oshape[-1])
    net = mx.sym.Deconvolution(data,
                               kernel=kshape,
                               stride=stride,
                               target_shape=target_shape,
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


def conv2d_bn_leaky(data, prefix, use_global_stats=False, **kwargs):
    net = mx.sym.Convolution(data, name="%s_conv" % prefix, **kwargs)
    net = BatchNorm(net, fix_gamma=True, eps=eps,
                    use_global_stats=use_global_stats,
                    name="%s_bn" % prefix)
    net = mx.sym.LeakyReLU(net, act_type="leaky", name="%s_leaky" % prefix)
    return net


def minibatch_layer(data, batch_size, num_kernels, num_dim=5):
    net = mx.sym.FullyConnected(data,
                                num_hidden=num_kernels*num_dim,
                                no_bias=True)
    net = mx.sym.Reshape(net, shape=(-1, num_kernels, num_dim))
    a = mx.sym.expand_dims(net, axis=3)
    b = mx.sym.expand_dims(
        mx.sym.transpose(net, axes=(1, 2, 0)), axis=0)
    abs_dif = mx.sym.abs(mx.sym.broadcast_minus(a, b))
    # batch, num_kernels, batch
    abs_dif = mx.sym.sum(abs_dif, axis=2)
    mask = np.eye(batch_size)
    mask = np.expand_dims(mask, 1)
    mask = 1.0 - mask
    rscale = 1.0 / np.sum(mask)
    # multiply by mask and rescale
    out = mx.sym.sum(mx.sym.broadcast_mul(abs_dif, constant(mask)), axis=2) * rscale

    return mx.sym.Concat(data, out)

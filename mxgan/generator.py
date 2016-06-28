"""Collection of generators symbols.

A generator takes random inputs and generate
"""
import numpy as np
import mxnet as mx

from .ops import deconv2d_bn_relu, deconv2d_act


def dcgan32x32(oshape, final_act, ngf=128, code=None):
    """DCGAN that generates 32x32 images."""
    assert oshape[-1] == 32
    assert oshape[-2] == 32
    code = mx.sym.Variable("code") if code is None else code
    net = mx.sym.FullyConnected(code, name="g1", num_hidden=4*4*ngf*4, no_bias=True)
    net = mx.sym.Activation(net, name="gact1", act_type="relu")
    # 4 x 4
    net = mx.sym.Reshape(net, shape=(-1, ngf * 4, 4, 4))
    # 8 x 8
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 4, 4, 4), oshape=(ngf * 2, 8, 8), kshape=(4, 4), prefix="g2")
    # 16x16
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 2, 8, 8), oshape=(ngf, 16, 16), kshape=(4, 4), prefix="g3")
    # 32x32
    net = deconv2d_act(
        net, ishape=(ngf, 16, 16), oshape=oshape[-3:], kshape=(4, 4), prefix="g4", act_type=final_act)
    return net


def dcgan28x28(oshape, final_act, ngf=128, code=None):
    """DCGAN that generates 28x28 images."""
    assert oshape[-1] == 28
    assert oshape[-2] == 28
    code = mx.sym.Variable("code") if code is None else code
    net = mx.sym.FullyConnected(code, name="g1", num_hidden=4*4*ngf*4, no_bias=True)
    net = mx.sym.Activation(net, name="gact1", act_type="relu")
    # 4 x 4
    net = mx.sym.Reshape(net, shape=(-1, ngf*4, 4, 4))
    # 8 x 8
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 4, 4, 4), oshape=(ngf * 2, 8, 8), kshape=(3, 3), prefix="g2")
    # 14x14
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 2, 8, 8), oshape=(ngf, 14, 14), kshape=(4, 4), prefix="g3")
    # 28x28
    net = deconv2d_act(
        net, ishape=(ngf, 14, 14), oshape=oshape[-3:], kshape=(4, 4), prefix="g4", act_type=final_act)
    return net


def fcgan(oshape, final_act, code=None):
    """DCGAN that generates 28x28 images using fully connected nets"""
    # Q: whether add BN
    code = mx.sym.Variable("code") if code is None else code
    net = mx.sym.FullyConnected(code, name="g1", num_hidden=500, no_bias=True)
    net = mx.sym.Activation(net, name="a1")
    net = mx.sym.FullyConnected(net2, name="g2", num_hidden=500, no_bias=True)
    net = mx.sym.Activation(net, name="a2")
    s = oshape[-3:]
    net = mx.sym.FullyConnected(net2, name="g3",
                                num_hidden=(s[-3] * s[-2] * s[-1]),
                                no_bias=True)
    net = mx.sym.Activation(net, name="gout", act_type=final_act)
    return net

"""Modules for training GAN, work with multiple GPU."""
import mxnet as mx
from . import ops
import numpy as np

class GANBaseModule(object):
    """Base class to hold gan data
    """
    def __init__(self,
                 symbol_generator,
                 context,
                 code_shape):
        # generator
        self.modG = mx.mod.Module(symbol=symbol_generator,
                                  data_names=("code",),
                                  label_names=None,
                                  context=context)
        self.modG.bind(data_shapes=[("code", code_shape)])
        # leave the discriminator
        self.temp_outG = None
        self.temp_diffD = None
        self.temp_gradD = None
        self.context = context if isinstance(context, list) else [context]
        self.outputs_fake = None
        self.outputs_real = None
        self.temp_rbatch = mx.io.DataBatch(
            [mx.nd.zeros(code_shape, ctx=self.context[-1])], None)

    def _save_temp_gradD(self):
        if self.temp_gradD is None:
            self.temp_gradD = [
                [grad.copyto(grad.context) for grad in grads]
                for grads in self.modD._exec_group.grad_arrays]
        else:
            for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr.copyto(gradf)

    def _add_temp_gradD(self):
        # add back saved gradient
        for gradsr, gradsf in zip(self.modD._exec_group.grad_arrays, self.temp_gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf

    def init_params(self, *args, **kwargs):
        self.modG.init_params(*args, **kwargs)
        self.modD.init_params(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self.modG.init_optimizer(*args, **kwargs)
        self.modD.init_optimizer(*args, **kwargs)


class GANModule(GANBaseModule):
    """A thin wrapper of module to group generator and discriminator together in GAN.

    Example
    -------
    lr = 0.0005
    mod = GANModule(generator, encoder, context=mx.gpu()),
    mod.bind(data_shape=(3, 32, 32))
    mod.init_params(mx.init.Xavier())
    mod.init_optimizer("adam", optimizer_params={
        "learning_rate": lr,
    })

    for t, batch in enumerate(train_data):
        mod.update(batch)
        # update metrics
        mod.temp_label[:] = 0.0
        metricG.update_metric(mod.outputs_fake, [mod.temp_label])
        mod.temp_label[:] = 1.0
        metricD.update_metric(mod.outputs_real, [mod.temp_label])
        # visualize
        if t % 100 == 0:
            gen_image = mod.temp_outG[0].asnumpy()
            gen_diff = mod.temp_diffD[0].asnumpy()
            viz.imshow("gen_image", gen_image)
            viz.imshow("gen_diff", gen_diff)
    """
    def __init__(self,
                 symbol_generator,
                 symbol_encoder,
                 context,
                 data_shape,
                 code_shape,
                 pos_label=0.9):
        super(GANModule, self).__init__(
            symbol_generator, context, code_shape)
        context = context if isinstance(context, list) else [context]
        self.batch_size = data_shape[0]
        label_shape = (self.batch_size, )
        encoder = symbol_encoder
        encoder = mx.sym.FullyConnected(encoder, num_hidden=1, name="fc_dloss")
        encoder = mx.sym.LogisticRegressionOutput(encoder, name='dloss')
        self.modD = mx.mod.Module(symbol=encoder,
                                  data_names=("data",),
                                  label_names=("dloss_label",),
                                  context=context)
        self.modD.bind(data_shapes=[("data", data_shape)],
                       label_shapes=[("dloss_label", label_shape)],
                       inputs_need_grad=True)
        self.pos_label = pos_label
        self.temp_label = mx.nd.zeros(
            label_shape, ctx=context[-1])

    def update(self, dbatch):
        """Update the model for a single batch."""
        # generate fake image
        mx.random.normal(0, 1.0, out=self.temp_rbatch.data[0])
        self.modG.forward(self.temp_rbatch)
        outG = self.modG.get_outputs()
        self.temp_label[:] = 0
        self.modD.forward(mx.io.DataBatch(outG, [self.temp_label]), is_train=True)
        self.modD.backward()
        self._save_temp_gradD()
        # update generator
        self.temp_label[:] = 1
        self.modD.forward(mx.io.DataBatch(outG, [self.temp_label]), is_train=True)
        self.modD.backward()
        diffD = self.modD.get_input_grads()
        self.modG.backward(diffD)
        self.modG.update()
        self.outputs_fake = [x.copyto(x.context) for x in self.modD.get_outputs()]
        # update discriminator
        self.temp_label[:] = self.pos_label
        dbatch.label = [self.temp_label]
        self.modD.forward(dbatch, is_train=True)
        self.modD.backward()
        self._add_temp_gradD()
        self.modD.update()
        self.outputs_real = self.modD.get_outputs()
        self.temp_outG = outG
        self.temp_diffD = diffD


class SemiGANModule(GANBaseModule):
    """A semisupervised gan that can take both labeled and unlabeled data.
    """
    def __init__(self,
                 symbol_generator,
                 symbol_encoder,
                 context,
                 data_shape,
                 code_shape,
                 num_class,
                 pos_label=0.9):
        super(SemiGANModule, self).__init__(
            symbol_generator, context, code_shape)
        # the discriminator encoder
        context = context if isinstance(context, list) else [context]
        batch_size = data_shape[0]
        self.num_class = num_class
        encoder = symbol_encoder
        encoder = mx.sym.FullyConnected(
            encoder, num_hidden=num_class + 1, name="energy")
        self.modD = mx.mod.Module(symbol=encoder,
                                  data_names=("data",),
                                  label_names=None,
                                  context=context)
        self.modD.bind(data_shapes=[("data", data_shape)],
                       inputs_need_grad=True)
        self.pos_label = pos_label
        # discriminator loss
        energy = mx.sym.Variable("energy")
        label_out = mx.sym.SoftmaxOutput(energy, name="softmax")
        ul_pos_energy = mx.sym.slice_axis(
            energy, axis=1, begin=0, end=num_class)
        ul_pos_energy = ops.log_sum_exp(
            ul_pos_energy, axis=1, keepdims=True, name="ul_pos")
        ul_neg_energy = mx.sym.slice_axis(
            energy, axis=1, begin=num_class, end=num_class + 1)
        ul_pos_prob = mx.sym.LogisticRegressionOutput(
            ul_pos_energy - ul_neg_energy, name="dloss")
        # use module to bind the
        self.mod_label_out = mx.mod.Module(
            symbol=label_out,
            data_names=("energy",),
            label_names=("softmax_label",),
            context=context)
        self.mod_label_out.bind(
            data_shapes=[("energy", (batch_size, num_class + 1))],
            label_shapes=[("softmax_label", (batch_size,))],
            inputs_need_grad=True)
        self.mod_ul_out = mx.mod.Module(
            symbol=ul_pos_prob,
            data_names=("energy",),
            label_names=("dloss_label",),
            context=context)
        self.mod_ul_out.bind(
            data_shapes=[("energy", (batch_size, num_class + 1))],
            label_shapes=[("dloss_label", (batch_size,))],
            inputs_need_grad=True)
        self.mod_ul_out.init_params()
        self.mod_label_out.init_params()
        self.temp_label = mx.nd.zeros(
            (batch_size,), ctx=context[0])

    def update(self, dbatch, is_labeled):
        """Update the model for a single batch."""
        # generate fake image
        mx.random.normal(0, 1.0, out=self.temp_rbatch.data[0])
        self.modG.forward(self.temp_rbatch)
        outG = self.modG.get_outputs()
        self.temp_label[:] = self.num_class
        self.modD.forward(mx.io.DataBatch(outG, []), is_train=True)
        self.mod_label_out.forward(
            mx.io.DataBatch(self.modD.get_outputs(), [self.temp_label]), is_train=True)
        self.mod_label_out.backward()
        self.modD.backward(self.mod_label_out.get_input_grads())
        self._save_temp_gradD()
        # update generator
        self.temp_label[:] = 1
        self.modD.forward(mx.io.DataBatch(outG, []), is_train=True)
        self.mod_ul_out.forward(
            mx.io.DataBatch(self.modD.get_outputs(), [self.temp_label]), is_train=True)
        self.mod_ul_out.backward()
        self.modD.backward(self.mod_ul_out.get_input_grads())
        diffD = self.modD.get_input_grads()
        self.modG.backward(diffD)
        self.modG.update()
        self.outputs_fake = [x.copyto(x.context) for x in self.mod_ul_out.get_outputs()]
        # update discriminator
        self.modD.forward(mx.io.DataBatch(dbatch.data, []), is_train=True)
        outD = self.modD.get_outputs()
        self.temp_label[:] = self.pos_label
        self.mod_ul_out.forward(
            mx.io.DataBatch(outD, [self.temp_label]), is_train=True)
        self.outputs_real = [x.copyto(x.context) for x in self.mod_ul_out.get_outputs()]
        if is_labeled:
            self.mod_label_out.forward(
                mx.io.DataBatch(outD, dbatch.label), is_train=True)
            self.mod_label_out.backward()
            egrad = self.mod_label_out.get_input_grads()
        else:
            self.mod_ul_out.backward()
            egrad = self.mod_ul_out.get_input_grads()
        self.modD.backward(egrad)
        self._add_temp_gradD()
        self.modD.update()
        self.temp_outG = outG
        self.temp_diffD = diffD

"""Modules for training GAN, work with multiple GPU."""
import mxnet as mx

class GANModule(object):
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
        self.batch_size = data_shape[0]
        # generator
        self.modG = mx.mod.Module(symbol=symbol_generator,
                                  data_names=("code",),
                                  label_names=None,
                                  context=context)
        # discriminator
        encoder, label_shape = self._init_classifier(symbol_encoder)
        self.modD = mx.mod.Module(symbol=encoder,
                                  data_names=("data",),
                                  label_names=("dloss_label",),
                                  context=context)
        self.modD.bind(data_shapes=[("data", data_shape)],
                       label_shapes=[("dloss_label", label_shape)],
                       inputs_need_grad=True)
        self.temp_label = None
        self.temp_outG = None
        self.temp_diffD = None
        self.temp_gradD = None
        self.context = context if isinstance(context, list) else [context]
        self.pos_label = pos_label
        self.outputs_fake = None
        self.outputs_real = None
        self.modG.bind(data_shapes=[("code", code_shape)])
        self.temp_label = mx.nd.zeros(label_shape, ctx=self.context[0])
        self.temp_rbatch = mx.io.DataBatch(
            [mx.nd.zeros(code_shape, ctx=self.context[-1])], None)

    def _init_classifier(self, encoder):
        encoder = mx.sym.FullyConnected(encoder, num_hidden=1, name="fc_dloss")
        encoder = mx.sym.LogisticRegressionOutput(encoder, name='dloss')
        return encoder, (self.batch_size, )

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

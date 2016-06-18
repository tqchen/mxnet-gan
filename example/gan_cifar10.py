import logging
import numpy as np
import mxnet as mx
import sys

sys.path.append("..")

from mxgan import module, generator, encoder, viz

def ferr(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return np.abs(label - (pred > 0.5)).sum() / label.shape[0]

ngf= 64
lr = 0.0003
beta1 = 0.5
batch_size = 100
rand_shape = (batch_size, 100)
num_epoch = 100
data_shape = (batch_size, 3, 32, 32)
context = mx.gpu()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
sym_gen = generator.dcgan32x32(oshape=data_shape, ngf=ngf, final_act="tanh")
sym_dec = encoder.dcgan(ngf=ngf / 2)
gmod = module.GANModule(
    sym_gen,
    sym_dec,
    context=context,
    data_shape=data_shape,
    code_shape=rand_shape)

gmod.modG.init_params(mx.init.Normal(0.05))
gmod.modD.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))

gmod.init_optimizer(
    optimizer="adam",
    optimizer_params={
        "learning_rate": lr,
        "wd": 0.,
        "beta1": beta1,
})

data_dir = './../../mxnet/example/image-classification/cifar10/'
train = mx.io.ImageRecordIter(
    path_imgrec = data_dir + "train.rec",
    data_shape = data_shape[1:],
    batch_size = batch_size,
    shuffle=True)

metric_acc = mx.metric.CustomMetric(ferr)

for epoch in range(num_epoch):
    train.reset()
    metric_acc.reset()
    for t, batch in enumerate(train):
        batch.data[0] = batch.data[0] * (1.0 / 255.0) - 0.5
        gmod.update(batch)
        gmod.temp_label[:] = 0.0
        metric_acc.update([gmod.temp_label], gmod.outputs_fake)
        gmod.temp_label[:] = 1.0
        metric_acc.update([gmod.temp_label], gmod.outputs_real)

        if t % 50 == 0:
            logging.info("epoch: %d, iter %d, metric=%s", epoch, t, metric_acc.get())
            viz.imshow("gout", gmod.temp_outG[0].asnumpy() + 0.5 , 2, flip=True)
            diff = gmod.temp_diffD[0].asnumpy()
            diff = (diff - diff.mean()) / diff.std() + 0.5
            viz.imshow("diff", diff, flip=True)
            viz.imshow("data", batch.data[0].asnumpy() + 0.5, 2, flip=True)

import os
import warnings

import global_const as gc
import layers as l
import keras.models as km
import keras.optimizers as ko
import keras.utils.vis_utils as kuv
import paths as op
import data as dat


def save_model():
    file_nam = op.get_full_file_nam(gc.prj.files.model)
    gc.prj.model.nn.save(file_nam)


def get_model():
    # check if model exists
    file_nam = op.get_full_file_nam(gc.prj.files.model)
    if os.path.exists(file_nam):
        gc.prj.model.nn = km.load_model(file_nam)
        print('Model available and loaded')
    else:
        mk_model()
        compile_model()
        gen_model_image()
        hist = train_model()
        dat.plot_metrics(hist)


def configure_optimizer(opti, lr):
    if opti == 'sgd':
        return ko.SGD(lr=lr)
    warnings.warn('lr not implemented for opti other than SGD')
    return opti


def compile_model():
    opti = configure_optimizer(gc.prj.model.optimizer, gc.prj.model.lr)
    gc.prj.model.nn.compile(loss=gc.prj.model.loss_func, optimizer=opti, metrics=gc.prj.model.metrics)


def train_model():
    hist = gc.prj.model.nn.fit(gc.prj.data.mod_ip.train_ip, gc.prj.data.mod_ip.train_op, batch_size=gc.prj.train.batch_siz,
                     epochs=gc.prj.train.nb_epochs, verbose=2,
                     validation_data=(gc.prj.data.mod_ip.test_ip, gc.prj.data.mod_ip.test_op),
                     callbacks=gc.prj.model.callback)
    save_model()
    return hist


def evaluate_model_metrics():
    metrics = gc.prj.model.nn.evaluate(gc.prj.data.mod_ip.test_ip, gc.prj.data.mod_ip.test_op, verbose=2)
    metrics_dict = dict()
    for i, key in enumerate(gc.prj.model.metrics):
        metrics_dict[key] = metrics[i]
    return metrics_dict


def use_model():
    gc.prj.data.pred_op = gc.prj.model.nn.predict_classes(gc.prj.data.mod_ip.test_ip)


def mk_model():
    layers = gc.prj.model.layers
    l.set_layer_names(layers)
    gc.prj.model.nn = km.Sequential(layers)


def gen_model_image():
    file_nam = op.get_full_file_nam(gc.prj.files.model_summary)
    # model.summary(print_fn= lambda x: print(x + '\n'))  # replace print with print to file
    kuv.plot_model(gc.prj.model.nn, to_file=file_nam, show_shapes=True, show_layer_names=True, expand_nested=True)

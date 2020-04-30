import os
import warnings

import global_const as gc
import layers as l
import keras.utils as ku
import keras.models as km
import keras.optimizers as ko
import keras.utils.vis_utils as kuv
import paths as op
import data as dat
import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz):
        gc.init_test_session(lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz)
        self._proc_data()

    def _save_model(self):
        file_nam = op.get_full_file_nam(gc.prj.files.model)
        gc.prj.model.nn.save(file_nam)

    def _get_ip_data(self):
        x_train = gc.prj.data.actual_ip.train_ip
        y_train = gc.prj.data.actual_ip.train_op
        x_test = gc.prj.data.actual_ip.test_ip
        y_test = gc.prj.data.actual_ip.test_op
        return (x_train, y_train), (x_test, y_test)

    def _normalize_apply_1hot(self, x_train, x_test, y_train, y_test):
        x_train = x_train.astype('float32')  # convert data
        x_test = x_test.astype('float32')
        x_train /= 255  # normalize data
        x_test /= 255

        gc.prj.data.mod_ip.train_ip = x_train
        gc.prj.data.mod_ip.test_ip = x_test
        num_classes = gc.prj.data.nb_class
        gc.prj.data.mod_ip.train_op = ku.to_categorical(y_train, num_classes)
        gc.prj.data.mod_ip.test_op = ku.to_categorical(y_test, num_classes)

    @abstractmethod
    def _proc_data(self):
        pass

    def setup_model(self, layers):
        gc.prj.model.layers = layers
        # check if model exists
        file_nam = op.get_full_file_nam(gc.prj.files.model)
        if os.path.exists(file_nam):
            gc.prj.model.nn = km.load_model(file_nam)
            print('Model available and loaded')
        else:
            self._mk_model()
            self._compile_model()
            self._gen_model_image()
            hist = self._train_model()
            dat.plot_metrics(hist)

    def _mk_model(self):
        layers = gc.prj.model.layers
        l.set_layer_names(layers)
        gc.prj.model.nn = km.Sequential(layers)

    def _configure_optimizer(self, opti, lr):
        if opti == 'sgd':
            return ko.SGD(lr=lr)
        warnings.warn('lr not implemented for opti other than SGD')
        return opti

    def _compile_model(self):
        opti = self._configure_optimizer(gc.prj.model.optimizer, gc.prj.model.lr)
        gc.prj.model.nn.compile(loss=gc.prj.model.loss_func, optimizer=opti, metrics=gc.prj.model.metrics)

    def _gen_model_image(self):
        file_nam = op.get_full_file_nam(gc.prj.files.model_summary)
        kuv.plot_model(gc.prj.model.nn, to_file=file_nam, show_shapes=True, show_layer_names=True, expand_nested=True)

    def _train_model(self):
        hist = gc.prj.model.nn.fit(gc.prj.data.mod_ip.train_ip, gc.prj.data.mod_ip.train_op, batch_size=gc.prj.train.batch_siz,
                         epochs=gc.prj.train.nb_epochs, verbose=2,
                         validation_data=(gc.prj.data.mod_ip.test_ip, gc.prj.data.mod_ip.test_op),
                         callbacks=gc.prj.model.callback)
        self._save_model()
        return hist

    def evaluate_model_metrics(self):
        metrics = gc.prj.model.nn.evaluate(gc.prj.data.mod_ip.test_ip, gc.prj.data.mod_ip.test_op, verbose=2)
        metrics_dict = dict()
        for i, key in enumerate(gc.prj.model.metrics):
            metrics_dict[key] = metrics[i]
        return metrics_dict

    def use_model(self):
        gc.prj.data.pred_op = gc.prj.model.nn.predict_classes(gc.prj.data.mod_ip.test_ip)
        dat.gen_image_summary(3, 3)

    def deinit(self):
        gc.deinit_test_session()


class ANN(Model):
    def __init__(self, lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz):
        super().__init__(lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz)

    def _proc_data(self):
        ((x_train, y_train), (x_test, y_test)) = self._get_ip_data()

        img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]
        x_train_flat = x_train.reshape(x_train.shape[0], img_rows * img_cols)
        x_test_flat = x_test.reshape(x_test.shape[0], img_rows * img_cols)
        self._normalize_apply_1hot(x_train=x_train_flat, x_test=x_test_flat, y_train=y_train, y_test=y_test)


class CNN(Model):
    def __init__(self, lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz):
        super().__init__(lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz)

    def _proc_data(self):
        ((x_train, y_train), (x_test, y_test)) = self._get_ip_data()

        img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]
        x_train_pix = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test_pix = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        self._normalize_apply_1hot(x_train=x_train_pix, x_test=x_test_pix, y_train=y_train, y_test=y_test)
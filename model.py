import os
import warnings

import keras.utils as ku
import keras.models as km
import keras.optimizers as ko
import keras.utils.vis_utils as kuv
import numpy as np
import keras.backend as kb
import gc as pgc

import obj as o
import utils as ut
import layers as l
import paths as op
import data as dat

from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        self.cfg = o.Cfg(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        dat.get_mnist_data(self.cfg)
        self._proc_data()

    def _save_model(self):
        file_nam = op.get_full_file_nam(self.cfg, self.cfg.prj.files.model)
        self.cfg.prj.model.nn.save(file_nam)

    def _normalize_apply_1hot(self, x_train, x_test, y_train, y_test):
        x_train = x_train.astype('float32')  # convert data
        x_test = x_test.astype('float32')
        x_train /= 255  # normalize data
        x_test /= 255

        num_classes = self.cfg.get_nb_classes()

        y_train = ku.to_categorical(y_train, num_classes)
        y_test = ku.to_categorical(y_test, num_classes)
        self.cfg.set_mod_ip(x_train, y_train, x_test, y_test)

    @abstractmethod
    def _proc_data(self):
        pass

    def setup_model(self, layers):
        self.cfg.set_layers(layers)
        l.set_layer_names(layers)
        # check if model exists
        file_nam = op.get_full_file_nam(self.cfg, self.cfg.prj.files.model)
        if os.path.exists(file_nam):
            self.cfg.prj.model.nn = km.load_model(file_nam)
        else:
            self._mk_model()
            self._compile_model()
            self._gen_model_image()
            self.cfg.set_eval_metrics(self._train_model())
            dat.plt_metrics(self.cfg)

    def _mk_model(self):
        layers = self.cfg.get_layers()
        self.cfg.prj.model.nn = km.Sequential(layers)

    def _configure_optimizer(self, opti, lr):
        if opti == 'sgd':
            return ko.SGD(lr=lr)
        warnings.warn('lr not implemented for opti other than SGD')
        return opti

    def _compile_model(self):
        opti = self._configure_optimizer(self.cfg.prj.model.optimizer, self.cfg.prj.model.lr)
        self.cfg.get_nn().compile(loss=self.cfg.prj.model.loss_func, optimizer=opti,
                                  metrics=self.cfg.prj.model.metrics)

    def _gen_model_image(self):
        file_nam = op.get_full_file_nam(self.cfg, self.cfg.prj.files.model_summary)
        kuv.plot_model(self.cfg.get_nn(), to_file=file_nam, show_shapes=True, show_layer_names=True,
                       expand_nested=True)

    def _train_model(self):
        ((x_train, y_train), (x_test, y_test)) = self.cfg.get_training_ip_data()
        batch, epoch = self.cfg.get_train_params()
        hist = self.cfg.get_nn().fit(x_train, y_train, batch_size=batch, epochs=epoch, verbose=2,
                                     validation_data=(x_test, y_test), callbacks=None)
        self._save_model()
        return hist

    def use_model(self):
        (x_test, y_test) = self.cfg.get_test_ip_data()

        metrics = self.cfg.get_nn().evaluate(x_test, y_test, verbose=2)
        self.cfg.set_test_metrics_result(metrics)

        y_pred_probab = self.cfg.get_nn().predict(x_test)
        self.cfg.set_test_probab_result(y_pred_probab)

        dat.gen_image_summary(self.cfg, 3, 3)
        dat.build_conf_matrix(self.cfg)
        dat.plt_roc(self.cfg)
        dat.plt_precision_recall_curve(self.cfg)

    def deinit(self):
        ut.dump(self.cfg)
        kb.clear_session()
        self.cfg.deinit()
        pgc.collect()


class ANN(Model):
    def __init__(self, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        super().__init__(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)

    def _proc_data(self):
        ((x_train, y_train), (x_test, y_test)) = self.cfg.get_ip_data()

        img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]
        x_train_flat = x_train.reshape(x_train.shape[0], img_rows * img_cols)
        x_test_flat = x_test.reshape(x_test.shape[0], img_rows * img_cols)

        self._normalize_apply_1hot(x_train=x_train_flat, x_test=x_test_flat, y_train=y_train, y_test=y_test)


class CNN(Model):
    def __init__(self, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        super().__init__(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)

    def _proc_data(self):
        ((x_train, y_train), (x_test, y_test)) = self.cfg.get_ip_data()

        img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]
        x_train_pix = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test_pix = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        self._normalize_apply_1hot(x_train=x_train_pix, x_test=x_test_pix, y_train=y_train, y_test=y_test)
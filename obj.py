from dataclasses import dataclass
from enum import Enum

import numpy as np
import gc as pygc

from keras.models import Model


@dataclass
class Files:
    op_dir: str = ''
    model: str = ''
    model_summary: str = ''
    metrics_plot: str = ''
    correct_image: str = ''
    incorrect_image: str = ''
    conf_matrix: str = ''
    roc_curve: str = ''
    prec_recall_curve: str = ''
    dump: str = ''
    image_lst: str = ''
# error_rate -> separate or within metrics plot?


@dataclass
class IpDataSet:
    train_ip: np.ndarray = None
    train_op: np.ndarray = None
    test_ip: np.ndarray = None
    test_op: np.ndarray = None


@dataclass
class OpDataSet:
    pred_op_prob: np.ndarray = None
    hist: np.ndarray = None
    metrics: list = None


@dataclass
class Data:
    actual_ip: IpDataSet = IpDataSet()
    mod_ip: IpDataSet = IpDataSet()
    out: OpDataSet = OpDataSet()
    nb_class: int = 0


class LossFunc(Enum):
    categorical_crossentropy = 'categorical_crossentropy'


class Optimizer(Enum):
    adam = 'adam'
    sgd = 'sgd'
    RMSprop = 'RMSprop'
    Adagrad = 'Adagrad'
    Adadelta = 'Adadelta'
    Adam = 'Adam'
    Adamax = 'Adamax'
    Nadam = 'Nadam'


class Metrics(Enum):
    loss = 'loss'
    accuracy = 'accuracy'


class Activation(Enum):
    relu = 'relu'
    linear = 'linear'
    softmax = 'softmax'
    sigmoid = 'sigmoid'
    hard_sigmoid = 'hard_sigmoid'
    tanh = 'tanh'
    exponential = 'exponential'


@dataclass
class Model:
    nn: Model = None
    layers: list = None
    loss_func: LossFunc = None
    optimizer: Optimizer = None
    metrics: list = None
    activation: Activation = None
    lr: float = 0.0


@dataclass
class Train:
    nb_epochs: int = 0
    batch_siz: int = 0


@dataclass
class GPUInfo:
    driver_ver: str = ''
    nb_gpu: int = 0
    gpu_name: str = ''
    gpu_mem_mb: float = 0.0
    gpu_clk_mhz: float = 0.0


@dataclass
class Misc:
    gpu_info: GPUInfo = GPUInfo()


@dataclass
class Prj:
    files: Files = Files()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()
    misc: Misc = Misc()


class Cfg:
    def __init__(self, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        self.prj = Prj()
        self._init_test_session(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)

    def _init_const_part(self):
        self.prj.files.model = 'model.h5'
        self.prj.files.model_summary = 'model.png'
        self.prj.files.op_dir = 'op'
        self.prj.files.correct_image = 'correct.png'
        self.prj.files.incorrect_image = 'incorrect.png'
        self.prj.files.metrics_plot = 'metrics.png'
        self.prj.files.conf_matrix = 'cmatrix.png'
        self.prj.files.roc_curve = 'roc_curve.png'
        self.prj.files.prec_recall_curve = 'precision_recall_curve.png'
        self.prj.files.image_lst = 'list.adoc'
        self.prj.files.dump = 'dump.json'
        self.prj.model.metrics = ['categorical_accuracy']

    def _init_test_session(self, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        self._init_const_part()
        self.prj.model.lr = lr

        self._validate_ele(Optimizer, optimizer)
        self.prj.model.optimizer = optimizer
        self._validate_ele(LossFunc, loss_func)
        self.prj.model.loss_func = loss_func
        self._validate_ele(Activation, activation)
        self.prj.model.activation = activation

        self.prj.train.nb_epochs = nb_epochs
        self.prj.train.batch_siz = batch_siz

    def _validate_ele(self, enum, ele):
        if enum.__name__ + '.' + ele not in list(map(str, enum)):
            raise ValueError(ele + ' not exist in ' + enum.__name__)

    def set_actual_ip(self, x_train, y_train, x_test, y_test, nb_classes):
        self.prj.data.actual_ip.train_ip = x_train
        self.prj.data.actual_ip.test_ip = x_test
        self.prj.data.actual_ip.train_op = y_train
        self.prj.data.actual_ip.test_op = y_test
        self.prj.data.nb_class = nb_classes

    def get_ip_data(self):
        x_train = self.prj.data.actual_ip.train_ip
        y_train = self.prj.data.actual_ip.train_op
        x_test = self.prj.data.actual_ip.test_ip
        y_test = self.prj.data.actual_ip.test_op
        return (x_train, y_train), (x_test, y_test)

    def get_training_ip_data(self):
        x_train = self.prj.data.mod_ip.train_ip
        y_train = self.prj.data.mod_ip.train_op
        x_test = self.prj.data.mod_ip.test_ip
        y_test = self.prj.data.mod_ip.test_op
        return (x_train, y_train), (x_test, y_test)

    def get_test_ip_data(self):
        x_test = self.prj.data.mod_ip.test_ip
        y_test = self.prj.data.mod_ip.test_op
        return x_test, y_test

    def set_test_probab_result(self, res):
        self.prj.data.out.pred_op_prob = res

    def set_test_metrics_result(self, res):
        self.prj.data.out.metrics = res

    def get_test_eval_op_params(self):
        y_test = self.prj.data.actual_ip.test_op
        y_pred = self.prj.data.out.pred_op_prob
        return y_test, y_pred

    def get_test_eval_params(self):
        x_test = self.prj.data.actual_ip.test_ip
        y_test, y_pred = self.get_test_eval_op_params()
        return x_test, y_test, y_pred

    def get_train_test_metrics(self):
        return self.prj.data.out.hist

    def set_eval_metrics(self, hist):
        self.prj.data.out.hist = hist

    def set_mod_ip(self, x_train, y_train, x_test, y_test):
        self.prj.data.mod_ip.train_ip = x_train
        self.prj.data.mod_ip.test_ip = x_test
        self.prj.data.mod_ip.train_op = y_train
        self.prj.data.mod_ip.test_op = y_test

    def get_nb_classes(self):
        return self.prj.data.nb_class

    def get_train_params(self):
        return self.prj.train.batch_siz, self.prj.train.nb_epochs

    def get_nn(self):
        return self.prj.model.nn

    def get_activation(self):
        return self.prj.model.activation

    def set_layers(self, layer_lst):
        self.prj.model.layers = layer_lst

    def get_layers(self):
        return self.prj.model.layers

    def set_gpu_info(self, driver_ver, gpu_cnt, gpu_name, gpu_mem_mb, clk_info):
        self.prj.misc.gpu_info.driver_ver = driver_ver
        self.prj.misc.gpu_info.nb_gpu = gpu_cnt
        self.prj.misc.gpu_info.gpu_name = gpu_name
        self.prj.misc.gpu_info.gpu_mem_mb = gpu_mem_mb
        self.prj.misc.gpu_info.clk_info = clk_info

    def deinit(self):
        try:
            del self.prj
            self.prj = None
        except:
            pass
        pygc.collect()

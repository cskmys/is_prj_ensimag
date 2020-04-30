from dataclasses import dataclass
from enum import Enum

import numpy as np

from keras.callbacks import Callback
from keras.models import Model


@dataclass
class Files:
    op_dir: str = ''
    model: str = ''
    model_summary: str = ''
    metrics_plot: str = ''
    correct_image: str = ''
    incorrect_image: str = ''
    pickle: str = ''
# roc_plot
# confusion_matrix
# precision_recall
# error_rate -> separate or within metrics plot?


@dataclass
class IpDataSet:
    train_ip: np.ndarray = None
    train_op: np.ndarray = None
    test_ip: np.ndarray = None
    test_op: np.ndarray = None


@dataclass
class Data:
    actual_ip: IpDataSet = IpDataSet()
    mod_ip: IpDataSet = IpDataSet()
    pred_op: np.ndarray = None
    nb_class: int = 0


class LossFunc(Enum):
    categorical_crossentropy = 'categorical_crossentropy'


class Optimizer(Enum):
    adam = 'adam'
    sgd = 'sgd'


class Metrics(Enum):
    loss = 'loss'
    accuracy = 'accuracy'


class Activation(Enum):
    relu = 'relu'
    linear = 'linear'
    softmax = 'softmax'


@dataclass
class Model:
    nn: Model = None
    layers: list = None
    loss_func: LossFunc = None
    optimizer: Optimizer = None
    metrics: list = None
    activation: Activation = None
    lr: float = 0.0
    callback: Callback = None


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

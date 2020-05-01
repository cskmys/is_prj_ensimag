from obj import Prj, Optimizer, LossFunc, Activation

import data as dat
import utils as ut
import keras.backend as kb
import gc as pgc

prj = None


def _init_const_part():
    global prj
    # initializing all those things that remain same throughout test cases
    prj.files.model = 'model.h5'
    prj.files.model_summary = 'model.png'
    prj.files.op_dir = 'op'
    prj.files.correct_image = 'correct.png'
    prj.files.incorrect_image = 'incorrect.png'
    prj.files.metrics_plot = 'metrics.png'
    prj.files.conf_matrix = 'cmatrix.png'

    prj.model.metrics = ['categorical_accuracy']
    # prj.model.lr
    # prj.model.optimizer
    # prj.model.loss_func
    # prj.model.activation
    # prj.model.callback
    # prj.train.nb_epochs
    # prj.train.batch_siz


def validate_ele(enum, ele):
    if enum.__name__ + '.' + ele not in list(map(str, enum)):
        raise ValueError( ele + ' not exist in ' + enum.__name__)


def init_test_session(lr, optimizer, loss_func, activation, callback, nb_epochs, batch_siz):
    global prj
    prj = Prj()
    _init_const_part()
    prj.model.lr = lr

    validate_ele(Optimizer, optimizer)
    prj.model.optimizer = optimizer
    validate_ele(LossFunc, loss_func)
    prj.model.loss_func = loss_func
    validate_ele(Activation, activation)
    prj.model.activation = activation

    prj.model.callback = callback
    prj.train.nb_epochs = nb_epochs
    prj.train.batch_siz = batch_siz


def deinit_test_session():
    kb.clear_session()
    try:
        global prj
        del prj
        prj = None
    except:
        pass
    pgc.collect()

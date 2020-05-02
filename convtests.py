import os

import matplotlib.pyplot as plt
import paths as op

import model as m
import layers as l
import utils as ut

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages

from time import time


def run_training(mod, layers):
    mod.setup_model(layers)


def run_validation_test(mod):
    mod.use_model()


def run_test(mod, layers):
    run_training(mod, layers)
    run_validation_test(mod)


def clean_up_test(mod):
    mod.deinit()


def run_test_suite(mod, layers):
    run_test(mod, layers)
    clean_up_test(mod)


def start_tst(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(cnn.cfg)
    layers = [L.ConvIp(2, (3, 3)), L.MaxPooling((2, 2)), L.Conv(2, (3, 3)), L.MaxPooling((2, 2)), L.Flat(),
              L.HiddenDense(128), L.DropOut(0.2), L.OpDense()]
    run_test_suite(cnn, layers)


def filter_tst_prev_model(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    for i in range(6):
        nb_filters = 2 ** (i+1)
        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(nb_filters, (3, 3)), L.MaxPooling((2, 2)), L.Conv(nb_filters, (3, 3)), L.MaxPooling((2, 2)), L.Flat(),
                  L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(cnn, layers)


def kernel_tst_prev_model(lr, optimizer, loss_func, activation, nb_epochs, batch_siz, filters):
    for i in range(4):
        kernel_dim = 3 + (2 * i)
        kernel = (kernel_dim, kernel_dim)
        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(filters, kernel), L.MaxPooling((2, 2)), L.Conv(filters, kernel), L.MaxPooling((2, 2)), L.Flat(),
                  L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(cnn, layers)


def kernel_pool_tst_prev_model(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(8, (3, 3)), L.MaxPooling((2, 2)), L.Conv(8, (3, 3)), L.MaxPooling((2, 2)), L.Flat(),
                  L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(cnn, layers)

        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(8, (3, 3)), L.AvgPooling(), L.Conv(8, (3, 3)), L.AvgPooling(), L.Flat(),
                  L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(cnn, layers)


def filter_tst(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    for i in range(6):
        nb_filters = 2 ** (i+1)
        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(nb_filters, (3, 3)), L.MaxPooling((2, 2)), L.Conv(nb_filters, (3, 3)), L.MaxPooling((2, 2)), L.Flat(),
                  L.HiddenDense(512), L.DropOut(0.2), L.OpDense()]
        run_test_suite(cnn, layers)


def kernel_tst(lr, optimizer, loss_func, activation, nb_epochs, batch_siz, filters):
    for i in range(4):
        kernel_dim = 3 + (2 * i)
        kernel = (kernel_dim, kernel_dim)
        cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(cnn.cfg)
        layers = [L.ConvIp(filters, kernel), L.MaxPooling((2, 2)), L.Conv(filters, kernel), L.MaxPooling((2, 2)), L.Flat(),
                  L.HiddenDense(512), L.DropOut(0.2), L.OpDense()]
        run_test_suite(cnn, layers)


def kernel_filter_test(units, lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    for i in range(6):
        nb_filters = 2 ** (i+1)
        for j in range(4):
            kernel_dim = 3 + (2 * j)
            kernel = (kernel_dim, kernel_dim)
            cnn = m.CNN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
            L = l.Layers(cnn.cfg)
            layers = [L.ConvIp(nb_filters, kernel), L.MaxPooling((2, 2)), L.Conv(nb_filters, kernel), L.MaxPooling((2, 2)), L.Flat(),
                      L.HiddenDense(units), L.DropOut(0.2), L.OpDense()]
            run_test_suite(cnn, layers)
            if j == 1:
                exit(0)


def test():
    # # start_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64)
    # # filter_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64)
    # # kernel_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=4)
    # kernel_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=8)
    # kernel_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=16)
    # kernel_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=32)
    # # kernel_tst(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=64)
    # kernel_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=2)
    # kernel_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=8)
    # kernel_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=16)
    # kernel_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=32)
    # # kernel_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64, filters=64)
    # kernel_filter_test(units=128, lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=10, batch_siz=128)
    kernel_pool_tst_prev_model(lr=0.002, optimizer='Adamax', loss_func='categorical_crossentropy', activation='relu', nb_epochs=6, batch_siz=64)
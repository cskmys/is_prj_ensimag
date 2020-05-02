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


def nb_layers_tst(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.OpDense()]
    run_test_suite(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
    run_test_suite(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
    run_test_suite(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.HiddenDense(512), L.HiddenDense(512), L.HiddenDense(512), L.OpDense()]
    run_test_suite(ann, layers)


def nb_neurons_tst(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(256), L.OpDense()]
    run_test_suite(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.OpDense()]
    run_test_suite(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(128), L.OpDense()]
    run_test_suite(ann, layers)


def nb_layers_with_epochs_tst(lr, optimizer, loss_func, nb_epochs, activation, batch_siz):
    for i in range(4):
        nb_layers_tst(lr, optimizer, loss_func, activation, nb_epochs ** i, batch_siz)


def nb_epochs_tst(lr, optimizer, loss_func, activation, batch_siz):
    metric_list = []
    nb_epochs_key = 'nb_epochs'
    plot_full_file_name = None
    plot_file_name = 'special.png'
    adoc_full_file_name = None
    for i in range(9):
        nb_epochs = 2 ** i
        ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(ann.cfg)
        layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
        run_test(ann, layers)
        res_dict = ut.get_res_dict(ann.cfg)
        res_dict[nb_epochs_key] = nb_epochs
        metric_list.append(res_dict)
        plot_full_file_name = op.get_full_file_nam(ann.cfg,
                                                   plot_file_name)  # will be saved in the output directory of last one
        adoc_full_file_name = op.get_full_file_nam(ann.cfg, ann.cfg.prj.files.image_lst)
        clean_up_test(ann)

    list_metrics = {k: [dic[k] for dic in metric_list] for k in metric_list[0]}
    epoch_list = list_metrics.pop(nb_epochs_key)

    fig = plt.figure()
    metrics = list_metrics.keys()
    for i, key in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        plt.plot(epoch_list, list_metrics[key])
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('epoch')
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(plot_full_file_name)
    adoc_op = ut.add_image_to_adoc('Metrics vs Epochs', plot_full_file_name, ut.get_adoc_tag(plot_file_name))
    with open(adoc_full_file_name, 'a+') as adoc_fil:
        adoc_fil.write(adoc_op)


def nb_batches_tst(lr, optimizer, loss_func, nb_epochs, activation):
    for i in range(9):
        batch_siz = 2 ** (i + 5)
        ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(ann.cfg)
        layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(ann, layers)


def nb_bacthes_epochs_tst(lr, optimizer, loss_func, activation):
    nb_batches_tst(lr, optimizer, loss_func, nb_epochs=16, activation=activation)
    nb_batches_tst(lr, optimizer, loss_func, nb_epochs=32, activation=activation)
    nb_batches_tst(lr, optimizer, loss_func, nb_epochs=64, activation=activation)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def activations_tst(lr, optimizer, loss_func, nb_epochs, batch_siz):
    activation_lst = ['relu', 'sigmoid', 'hard_sigmoid', 'tanh', 'linear']
    for act in activation_lst:
        ann = m.ANN(lr, optimizer, loss_func, act, nb_epochs, batch_siz)
        L = l.Layers(ann.cfg)
        layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
        tInit = time()
        run_training(ann, layers)
        tFinal = time()
        run_validation_test(ann)
        clean_up_test(ann)
        print(bcolors.OKGREEN + str(tFinal - tInit) + bcolors.ENDC)


def loss_func_tst(lr, loss_func, optimizer_lst, activation, nb_epochs, batch_siz):
    metrics_lst = []
    plot_full_file_name = None
    plot_file_name = 'special.png'
    adoc_full_file_name = None
    time_key = 'time'
    for optimizer in optimizer_lst:
        ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(ann.cfg)
        layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
        tInit = time()
        run_test(ann, layers)
        tFinal = time()
        print(bcolors.OKGREEN + str(tFinal - tInit) + bcolors.ENDC)
        res_dict = ut.get_res_dict(ann.cfg)
        res_dict[time_key] = tFinal - tInit
        metrics_lst.append(res_dict)
        plot_full_file_name = op.get_full_file_nam(ann.cfg,
                                                   plot_file_name)  # will be saved in the output directory of last one
        adoc_full_file_name = op.get_full_file_nam(ann.cfg, ann.cfg.prj.files.image_lst)
        clean_up_test(ann)

    plot_title = 'Optimizer and Model Accuracy'
    if os.path.isfile(plot_full_file_name):
        adoc_op = ut.add_image_to_adoc(plot_title, plot_full_file_name, ut.get_adoc_tag(plot_file_name))
        with open(adoc_full_file_name, 'a+') as adoc_fil:
            adoc_fil.write(adoc_op)
        return

    list_metrics = {k: [dic[k] for dic in metrics_lst] for k in metrics_lst[0]}
    time_val = list_metrics.pop(time_key)
    key = list(list_metrics.keys())
    x = list_metrics[key[0]]
    y = list_metrics[key[1]]
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(x, y)
    n = optimizer_lst
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], y[i]))
    plt.title(plot_title)
    plt.ylabel(key[1])
    plt.xlabel(key[0])

    plt.subplot(2, 1, 2)
    plt.bar(optimizer_lst, time_val)
    plt.ylabel(time_key)
    plt.close(fig)
    fig.savefig(plot_full_file_name)
    adoc_op = ut.add_image_to_adoc(plot_title, plot_full_file_name, ut.get_adoc_tag(plot_file_name))
    with open(adoc_full_file_name, 'a+') as adoc_fil:
        adoc_fil.write(adoc_op)


def test():
    # nb_neurons_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=4, batch_siz=8 * 1024)
    # nb_layers_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=4, batch_siz=8 * 1024)
    # nb_layers_with_epochs_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', nb_epochs=2, activation='relu', batch_siz=8 * 1024)
    # nb_epochs_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', batch_siz=8 * 1024)
    # nb_bacthes_epochs_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu')
    # activations_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', nb_epochs=32, batch_siz=64)
    # optimizer_lst = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # loss_func_tst(lr=0.01, loss_func='categorical_crossentropy', optimizer_lst=optimizer_lst, activation='relu', nb_epochs=32, batch_siz=64)
    # optimizer_lst = ['Adagrad', 'Adadelta', 'Adamax']
    # loss_func_tst(lr=0.01, loss_func='categorical_crossentropy', optimizer_lst=optimizer_lst, activation='relu',
    #               nb_epochs=5, batch_siz=64)

    # nb_neurons_tst(lr=0.002, loss_func='categorical_crossentropy', optimizer='Adamax', activation='relu',
    #                nb_epochs=5, batch_siz=64)
    nb_neurons_tst(lr=0.003, loss_func='categorical_crossentropy', optimizer='Adamax', activation='relu',
                    nb_epochs=5, batch_siz=64)
# for i in range(5):
#
#     ann = m.ANN(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=2, batch_siz=8 * 1024)
#     L = l.Layers(ann.cfg)
#     layers = [L.IpDense(512), L.DropOut((i + 1) / 20), L.HiddenDense(512), L.DropOut(0.2), L.OpDense()]
#     run_test(ann, layers)
#
#     cnn = m.CNN(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=2, batch_siz=8 * 1024)
#     L = l.Layers(cnn.cfg)
#     layers = [L.ConvIp(6, (3, 3)), L.AvgPooling(), L.Conv(16, (3, 3)), L.AvgPooling(), L.Flat(), L.HiddenDense(120),
#               L.HiddenDense(84), L.OpDense()]
#     run_test(cnn, layers)

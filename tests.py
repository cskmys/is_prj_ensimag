import os

import matplotlib.pyplot as plt
import paths as op

import model as m
import layers as l
import utils as ut

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages


def run_test(mod, layers):
    mod.setup_model(layers)
    mod.use_model()


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
        plot_full_file_name = op.get_full_file_nam(ann.cfg, plot_file_name)  # will be saved in the output directory of last one
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
        batch_siz = 2 ** (i+5)
        ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
        L = l.Layers(ann.cfg)
        layers = [L.IpDense(512), L.HiddenDense(512), L.OpDense()]
        run_test_suite(ann, layers)


def test():
    # nb_neurons_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=4, batch_siz=8 * 1024)
    # nb_layers_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=4, batch_siz=8 * 1024)
    # nb_layers_with_epochs_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', nb_epochs=2, activation='relu', batch_siz=8 * 1024)
    # nb_epochs_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', batch_siz=8 * 1024)
    # nb_batches_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy',  nb_epochs=16, activation='relu')
    nb_batches_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy',  nb_epochs=32, activation='relu')
    nb_batches_tst(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy',  nb_epochs=64, activation='relu')
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


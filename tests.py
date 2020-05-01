import os

import model as m
import layers as l

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages


def run_test(mod, layers):
    mod.setup_model(layers)
    mod.use_model()
    mod.deinit()


def simple_ann(lr, optimizer, loss_func, activation, nb_epochs, batch_siz):
    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.OpDense()]
    run_test(ann, layers)

    ann = m.ANN(lr, optimizer, loss_func, activation, nb_epochs, batch_siz)
    L = l.Layers(ann.cfg)
    layers = [L.IpDense(512), L.IpDense(512), L.OpDense()]
    run_test(ann, layers)


def test():
    simple_ann(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=2, batch_siz=8 * 1024)
    simple_ann(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', nb_epochs=2,
               batch_siz=4 * 1024)

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


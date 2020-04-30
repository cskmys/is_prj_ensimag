import os

import global_const as gc
import data as dat
import model as m
import layers as l

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages


def run_test(mod, layers):
    mod.setup_model(layers)
    mod.evaluate_model_metrics()
    mod.use_model()
    mod.deinit()


for i in range(5):

    ann = m.ANN(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', callback=None, nb_epochs=2, batch_siz=8 * 1024)
    layers = [l.IpDense(512), l.DropOut((i + 1) / 20), l.Dense(512), l.Dropout(0.2), l.OpDense()]
    run_test(ann, layers)
    # layers = [l.ConvIp(6, (3, 3)), l.AvgPooling(), l.Conv(16, (3, 3)), l.AvgPooling(), l.Flat(), l.Dense(120),
    #           l.Dense(84), l.OpDense()]

exit(0)

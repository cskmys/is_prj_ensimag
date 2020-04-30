import os

import global_const as gc
import data as dat
import model as m
import layers as l

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages

for i in range(5):
    gc.init_test_data(lr=0.01, optimizer='sgd', loss_func='categorical_crossentropy', activation='relu', callback=None, nb_epochs=2, batch_siz=8 * 1024)
    dat.proc_data()
    gc.prj.model.layers = [l.IpDense(512), l.DropOut((i+1)/20), l.Dense(512), l.Dropout(0.2), l.OpDense()]
    m.get_model()

    m.evaluate_model_metrics()
    m.use_model()
    dat.gen_image_summary(3, 3)

    gc.deinit_test_data()

exit(0)

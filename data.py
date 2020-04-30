import keras
import matplotlib
import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt
import global_const as gc
import paths as op

matplotlib.use('agg')


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = np.unique(y_test).size

    gc.prj.data.actual_ip.train_ip = x_train
    gc.prj.data.actual_ip.train_op = y_train
    gc.prj.data.actual_ip.test_ip = x_test
    gc.prj.data.actual_ip.test_op = y_test
    gc.prj.data.nb_class = num_classes


def proc_data():
    x_train = gc.prj.data.actual_ip.train_ip
    y_train = gc.prj.data.actual_ip.train_op
    x_test = gc.prj.data.actual_ip.test_ip
    y_test = gc.prj.data.actual_ip.test_op
    num_classes = gc.prj.data.nb_class

    img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]
    x_train_flat = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test_flat = x_test.reshape(x_test.shape[0], img_rows * img_cols)

    x_train_flat = x_train_flat.astype('float32')  # convert data
    x_test_flat = x_test_flat.astype('float32')
    x_train_flat /= 255  # normalize data
    x_test_flat /= 255

    y_train_1_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_1_hot = keras.utils.to_categorical(y_test, num_classes)

    gc.prj.data.mod_ip.train_ip = x_train_flat
    gc.prj.data.mod_ip.train_op = y_train_1_hot
    gc.prj.data.mod_ip.test_ip = x_test_flat
    gc.prj.data.mod_ip.test_op = y_test_1_hot


def plot_metrics(hist):
    fig = plt.figure()
    metrics = list(hist.history.keys())
    nb_metrics = len(metrics) // 2
    for i in range(nb_metrics):
        plt.subplot(nb_metrics, 1, i + 1)
        plt.plot(hist.history[metrics[i]])
        plt.plot(hist.history[metrics[i + nb_metrics]])
        plt.title('model ' + metrics[i + nb_metrics])
        plt.ylabel(metrics[i + nb_metrics])
        plt.xlabel('epoch')
        plt.legend(['test', 'train'], loc='best')
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(op.get_full_file_nam(gc.prj.files.metrics_plot))


# https://nextjournal.com/gkoehler/digit-recognition-with-keras
def gen_image(x, y_p, y_t, row, col, name, splt_print):
    fig = plt.figure()
    plt.title(name)
    for i in range(row * col):
        plt.subplot(row, col, i+1)
        plt.tight_layout()
        plt.imshow(x[i], cmap='gray', interpolation='none')
        splt_print(plt, x[i], y_p[i], y_t[i])
        plt.xticks([])
        plt.yticks([])
    plt.close(fig)
    return fig


def gen_correct_image(x, y_pred, y_test, row, col):
    correct_indices = np.nonzero(y_pred == y_test)[0]
    correct_x = []
    correct_y = []
    for i, correct in enumerate(correct_indices[:(row * col)]):
        correct_x.append(x[correct])
        correct_y.append(y_test[correct])
    return gen_image(correct_x, correct_y, correct_y, row, col, 'Correct',
                     lambda p, x_idx, yp_idx, yt_idx: p.title('Digit: {}'.format(yp_idx)))


def gen_incorrect_image(x, y_pred, y_test, row, col):
    incorrect_indices = np.nonzero(y_pred != y_test)[0]
    incorrect_x = []
    incorrect_y = []
    correct_y = []
    for i, incorrect in enumerate(incorrect_indices[:(row * col)]):
        incorrect_x.append(x[incorrect])
        incorrect_y.append(y_pred[incorrect])
        correct_y.append(y_test[incorrect])
    return gen_image(incorrect_x, incorrect_y, correct_y, row, col, 'Incorrect',
                     lambda p, x_idx, yp_idx, yt_idx: p.title('Pred: {} Actual: {}'.format(yp_idx, yt_idx)))


def gen_image_summary(row, col):
    assert gc.prj.data.pred_op is not None, "Run use_model 1st"
    corr_fig = gen_correct_image(gc.prj.data.actual_ip.test_ip, gc.prj.data.pred_op, gc.prj.data.actual_ip.test_op, row, col)
    incorr_fig = gen_incorrect_image(gc.prj.data.actual_ip.test_ip, gc.prj.data.pred_op, gc.prj.data.actual_ip.test_op, row, col)
    corr_fig.savefig(op.get_full_file_nam(gc.prj.files.correct_image))
    incorr_fig.savefig(op.get_full_file_nam(gc.prj.files.incorrect_image))


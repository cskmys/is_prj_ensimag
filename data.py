import matplotlib
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall

import paths as op

matplotlib.use('agg')


def get_mnist_data(cfg):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = np.unique(y_test).size
    cfg.set_actual_ip(x_train, y_train, x_test, y_test, num_classes)


def plt_metrics(cfg):
    fig = plt.figure()
    hist = cfg.get_train_test_metrics()
    metrics = list(hist.history.keys())
    nb_metrics = len(metrics) // 2
    for i in range(nb_metrics):
        plt.subplot(nb_metrics, 1, i + 1)
        plt.plot(hist.history[metrics[i + nb_metrics]])
        plt.plot(hist.history[metrics[i]])
        plt.title('model ' + metrics[i + nb_metrics])
        plt.ylabel(metrics[i + nb_metrics])
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(op.get_full_file_nam(cfg, cfg.prj.files.metrics_plot))


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


def image_gen_precheck(cfg):
    y_test, y_pred = cfg.get_test_eval_op_params()
    assert y_pred is not None, "Run use_model 1st"


def get_pred_class(y_pred_prob):
    y_pred_prob = np.asarray(y_pred_prob)
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    return y_pred_class


def gen_image_summary(cfg, row, col):
    image_gen_precheck(cfg)
    x_test, y_test, y_pred_prob = cfg.get_test_eval_params()
    y_pred_class = get_pred_class(y_pred_prob)
    corr_fig = gen_correct_image(x_test, y_pred_class, y_test, row, col)
    incorr_fig = gen_incorrect_image(x_test, y_pred_class, y_test, row, col)
    corr_fig.savefig(op.get_full_file_nam(cfg, cfg.prj.files.correct_image))
    incorr_fig.savefig(op.get_full_file_nam(cfg, cfg.prj.files.incorrect_image))


def build_conf_matrix(cfg):
    image_gen_precheck(cfg)
    y_test, y_pred_prob = cfg.get_test_eval_op_params()
    y_pred_class = get_pred_class(y_pred_prob)
    plot_confusion_matrix(y_test, y_pred_class, normalize=True, title='Normalized Confusion Matrix')
    plt.savefig(op.get_full_file_nam(cfg, cfg.prj.files.conf_matrix))
    plt.close()


def plt_roc(cfg):
    image_gen_precheck(cfg)
    y_test, y_pred_prob = cfg.get_test_eval_op_params()
    plot_roc(y_test, y_pred_prob)
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(op.get_full_file_nam(cfg, cfg.prj.files.roc_curve), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plt_precision_recall_curve(cfg):
    image_gen_precheck(cfg)
    y_test, y_pred_prob = cfg.get_test_eval_op_params()
    plot_precision_recall(y_test, y_pred_prob)
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(op.get_full_file_nam(cfg, cfg.prj.files.prec_recall_curve), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
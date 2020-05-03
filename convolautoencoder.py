from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages


def plt_metrics(histo, save_name):
    fig = plt.figure()
    metrics = list(histo.history.keys())
    nb_metrics = len(metrics) // 2
    for i in range(nb_metrics):
        plt.subplot(nb_metrics, 1, i + 1)
        plt.plot(histo.history[metrics[i + nb_metrics]])
        plt.plot(histo.history[metrics[i]])
        plt.title('model ' + metrics[i + nb_metrics])
        plt.ylabel(metrics[i + nb_metrics])
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='best')
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(save_name)


def plot_op(x_test, dec_imgs, save_name):
    n = 10  # how many digits we will display
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')

        # display reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(dec_imgs[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
    plt.close(fig)
    fig.savefig(save_name)


def get_full_file_name(file_name):
    return os.path.join('./op/Autoencoder', file_name)


def gen_model_img(model, file_name):
    plot_model(model, get_full_file_name(file_name), show_shapes=True, show_layer_names=True, expand_nested=True)


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

gen_model_img(autoencoder, 'convautoencoder.png')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

hist = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=2)
plt_metrics(hist, get_full_file_name('convautoencoder_metrics.png'))

decoded_imgs = autoencoder.predict(x_test)

plot_op(x_test, decoded_imgs, get_full_file_name('convautoencoder_op.png'))
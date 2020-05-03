from keras.layers import Input, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

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


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

gen_model_img(encoder, 'deepautoencoder_encoder.png')
gen_model_img(decoder, 'deepautoencoder_decoder.png')
gen_model_img(autoencoder, 'deepautoencoder.png')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

hist = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=2)
plt_metrics(hist, get_full_file_name('deepautoencoder_metrics.png'))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

plot_op(x_test, decoded_imgs, get_full_file_name('deepautoencoder_op.png'))
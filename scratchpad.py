from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages

num_classes = 10

batch_size = 1024 * 8  # 128
epochs = 1 # 24

img_rows, img_cols = 28, 28

(x_train, y_train), (X_test, Y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':  # ensure compatibility bw different backends. Some put the # of channels
    # in the image before the width and height of image
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# 1 -> nb channels in the image, here its greyscale image

x_train = x_train.astype('float32')  # convert data
x_test = x_test.astype('float32')
x_train /= 255  # normalize data
x_test /= 255

# convert class vectors to binary class matrices. One-hot encoding
#  3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
# output should not be a number but rather the neuron index that gets activated

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # convert 2-d output of image data into 1-d data for a conventional NN layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # to prevent network tendency to overfit the data
model.add(Dense(num_classes, activation='softmax'))

import tensorflow as tf
from keras import backend as K


# def auc1(y_true, y_pred):
#     auc = tf.metrics.AUC()
#     auc.update_state(y_true, y_pred)
#     return auc.result()

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
metrics_lst = ['accuracy', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()]
# metrics_lst = ['accuracy',auc1]
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=[metrics_lst])

model.summary()

#   Return history of loss and accuracy for each epoch
# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

from sklearn.metrics import roc_curve, plot_roc_curve
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import numpy as np
import matplotlib.pyplot as plt

class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_temp = self.model.predict(x_test)#self.validation_data[0])
        y_pred = np.asarray(y_temp)
        y_true = self.validation_data[1]

        y_pred_class = self.model.predict_classes(x_test) # np.argmax(y_pred, axis=1)
        y_test_class = Y_test  # np.argmax(y_true, axis=1)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_test_class, y_pred_class, ax=ax)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_test_class, y_pred, ax=ax)

        # y_pred_train = self.model.predict_proba(self.x)
        # roc_train = roc_auc_score(self.y, y_pred_train)
        # y_pred_val = self.model.predict_proba(self.x_val)
        # roc_val = roc_auc_score(self.y_val, y_pred_val)
        # print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


roc = RocCallback(training_data=(x_train, y_train),validation_data=(x_test, y_test))

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[roc])
score = model.evaluate(x_test, y_test, verbose=0)  # Evaluate model with test data to get scores on "real" data.
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# from sklearn.metrics import roc_curve, plot_roc_curve
# from scikitplot.metrics import plot_confusion_matrix, plot_roc
# y_pred = model.predict(x_test)
# plot_roc(y_test, y_pred, )
# import numpy as np
# fpr, tpr, thresholds = roc_curve(np.round(y_test), np.round(y_pred))



# Plot data to see relationships in training and validation data

epoch_list = list(range(1, len(hist.history['accuracy']) + 1))  # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()
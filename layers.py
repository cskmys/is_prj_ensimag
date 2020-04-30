from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D

import global_const as gc


def get_layer_name(layer):
    return layer.__class__.__name__


def get_layer_param_str(layer):
    layer_name = get_layer_name(layer)
    if layer_name == 'Dropout':
        return str(layer.rate)
    if layer_name == 'Dense':
        return layer.activation.__name__
    if layer_name == 'Conv2D':
        (x, y) = layer.kernel_size
        return str('{fil}f{x}x{y}').format(fil=layer.filters, x=x, y=y)
    if layer_name == 'MaxPooling2D':
        (x, y) = layer.pool_size
        return str('{x}x{y}').format(x=x, y=y)
    if layer_name == 'Flatten':
        return ''
    if layer_name == 'AveragePooling2D':
        return ''
    raise TypeError('Layer ' + layer.__class__.__name__ + ' still unsupported')


def set_name_for_layer(layers, layer_idx):
    layer = layers[layer_idx]
    layer_idx = layers.index(layer)
    name = 'l' + str(layer_idx) + '_' + get_layer_name(layer) + get_layer_param_str(layer)
    layers[layer_idx].name = name


def set_layer_names(layers):
    for i, layer in enumerate(layers):
        set_name_for_layer(layers, i)


def IpDense(units):
    return Dense(units=units, input_shape=(gc.prj.data.mod_ip.test_ip.shape[1], ), activation=gc.prj.model.activation)


def HiddenDense(units):
    return Dense(units=units, activation=gc.prj.model.activation)


def DropOut(rate):
    return Dropout(rate=rate)


def OpDense():
    return Dense(units=gc.prj.data.nb_class, activation='softmax')


def ConvIp(filter, kernel_shape):
    img_rows = gc.prj.data.actual_ip.train_ip.shape[1]
    img_cols = gc.prj.data.actual_ip.train_ip.shape[2]
    ip_shape = (img_rows, img_cols, 1)
    return Conv2D(filters=filter, kernel_size=kernel_shape, input_shape=ip_shape, activation=gc.prj.model.activation)


def Conv(filter, kernel_shape):
    return Conv2D(filters=filter, kernel_size=kernel_shape, activation=gc.prj.model.activation)


def MaxPooling(pool_shape):
    return MaxPooling2D(pool_size=pool_shape)


def Flat():
    return Flatten()


def AvgPooling():
    return AveragePooling2D()

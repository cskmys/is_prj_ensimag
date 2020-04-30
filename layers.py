from keras.layers.core import Dense, Dropout

import global_const as gc


def get_layer_param_str(layer):
    if hasattr(layer, 'rate'):
        return str(layer.rate)
    if hasattr(layer, 'activation'):
        return layer.activation.__name__
    return ''


def get_layer_name(layer):
    return layer.__class__.__name__


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

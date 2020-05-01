import os
from pathlib import Path

import layers as l


def name_layers(layers):
    name = str()
    for layer in layers:
        name = name + l.get_layer_name(layer) + l.get_layer_param_str(layer)
    return name


def _get_cur_sub_dir_name(cfg):
    model = cfg.prj.model
    train = cfg.prj.train
    l_hash = name_layers(model.layers)
    subdir = '{opti}{loss}{act}l{lr}e{epochs}b{batch}m{hash}'.format(opti=model.optimizer, loss=model.loss_func,
                                                                     act=model.activation, lr=model.lr,
                                                                     epochs=train.nb_epochs, batch=train.batch_siz,
                                                                     hash=l_hash)
    return subdir


def get_full_file_nam(cfg, file_nam):
    file_path = os.path.join('./', cfg.prj.files.op_dir, _get_cur_sub_dir_name(cfg))
    Path(file_path).mkdir(parents=True, exist_ok=True)

    full_file_nam = os.path.join(file_path, file_nam)
    assert len(full_file_nam) < 256, 'File name too big'
    return full_file_nam


def get_dump_file_name(cfg):
    file_name = os.path.join('./', cfg.prj.files.op_dir, cfg.prj.files.dump)
    return file_name

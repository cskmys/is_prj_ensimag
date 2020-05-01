import pynvml as nv
import layers as l
import paths as op

import json


def get_gpu_info(cfg):
    nv.nvmlInit()
    driver_ver = nv.nvmlSystemGetDriverVersion().decode()
    gpu_cnt = nv.nvmlDeviceGetCount()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    gpu_name = nv.nvmlDeviceGetName(handle).decode()
    mem_info = nv.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem_mb = mem_info.total / 1024 / 1024
    clk_info = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS)
    nv.nvmlShutdown()
    cfg.set_gpu_info(driver_ver, gpu_cnt, gpu_name, gpu_mem_mb, clk_info)


def get_layer_dump_lst(layers):
    layer_lst = list()
    for layer in layers:
        layer_lst.append(l.get_layer_name(layer) + ' ' + l.get_layer_param_str(layer))
    return layer_lst


def dump_already_exists(cfg):
    search_str = '"' + op.get_full_file_nam(cfg, './') + '",'
    print(search_str)
    with open('./op/dump.json', 'r', encoding='utf-8') as dump_file:
        if search_str in dump_file.read():
            return True
    return False

def dump_json(cfg):
    dump_dict = {
        'sub_dir': op.get_full_file_nam(cfg, './'),
        'layers': get_layer_dump_lst(cfg.prj.model.layers),
        'result_metrics': cfg.prj.data.out.metrics,
        'loss_func': cfg.prj.model.loss_func,
        'optimizer': cfg.prj.model.optimizer,
        'metrics': cfg.prj.model.metrics,
        'activation': cfg.prj.model.activation,
        'lr': cfg.prj.model.lr,
        'nb_epochs': cfg.prj.train.nb_epochs,
        'batch_size': cfg.prj.train.batch_siz
    }
    with open(op.get_dump_file_name(cfg), 'a', encoding='utf-8') as dump_file:
        json.dump(dump_dict, dump_file, ensure_ascii=False, indent=4)
        dump_file.write('\n')


def add_image_to_adoc(image_title, file):
    adoc_str = str('.{title}\n'
                   'image::{file}[{title}]\n').format(title=image_title, file=file)
    return adoc_str


def dump_adoc(cfg):
    img_op = str()
    adoc_str_lst = [('Confusion Matrix', op.get_full_file_nam(cfg, cfg.prj.files.conf_matrix)),
                    ('Correctly classified samples', op.get_full_file_nam(cfg, cfg.prj.files.correct_image)),
                    ('Incorrectly classified samples', op.get_full_file_nam(cfg, cfg.prj.files.incorrect_image)),
                    ('Metrics', op.get_full_file_nam(cfg, cfg.prj.files.metrics_plot)),
                    ('Model Summary', op.get_full_file_nam(cfg, cfg.prj.files.model_summary)),
                    ('Precision-Recall', op.get_full_file_nam(cfg, cfg.prj.files.prec_recall_curve)),
                    ('ROC', op.get_full_file_nam(cfg, cfg.prj.files.roc_curve)),
                    ]
    for title, file_name in adoc_str_lst:
        img_op += add_image_to_adoc(title, file_name)

    with open(op.get_full_file_nam(cfg, cfg.prj.files.image_lst), 'w') as adoc_fil:
        adoc_fil.write(img_op)

    with open(op.get_full_file_nam(cfg, cfg.prj.files.image_lst), 'a') as adoc_fil:
        adoc_fil.write('----\n')
        json.dump(get_layer_dump_lst(cfg.prj.model.layers), adoc_fil)
        adoc_fil.write('\n----\n')


def dump(cfg):
    if dump_already_exists(cfg) is False:
        dump_json(cfg)
        dump_adoc(cfg)

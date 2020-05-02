import pynvml as nv
import layers as l
import paths as op

import json
from pprint import pprint


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
    try:
        with open(op.get_full_file_nam(cfg, cfg.prj.files.dump), 'r', encoding='utf-8') as dump_file:
            if search_str in dump_file.read():
                return True
    except FileNotFoundError:
        return False
    return False


def dump_json(cfg):
    dump_dict = {
        'sub_dir': op.get_full_file_nam(cfg, './'),
        'layers': get_layer_dump_lst(cfg.prj.model.layers),
        'result_metrics': cfg.prj.data.out.metrics,  # for prediction after evaluation
        'loss_func': cfg.prj.model.loss_func,
        'optimizer': cfg.prj.model.optimizer,
        'metrics': cfg.prj.model.metrics,
        'activation': cfg.prj.model.activation,
        'lr': cfg.prj.model.lr,
        'nb_epochs': cfg.prj.train.nb_epochs,
        'batch_size': cfg.prj.train.batch_siz
    }
    jobj = []
    try:
        with open(op.get_dump_file_name(cfg), 'r', encoding='utf-8') as dump_file:
            jobj = json.load(dump_file)
            found = False
            for i, j in enumerate(jobj):
                if j['sub_dir'] == dump_dict['sub_dir']:
                    jobj[i]['result_metrics'] = dump_dict['result_metrics']  # changing only the dynamic part
                    found = True
                    break
            if found is False:
                jobj.append(dump_dict)
    except FileNotFoundError:
        jobj.append(dump_dict)
    pprint(dump_dict)
    with open(op.get_dump_file_name(cfg), 'w', encoding='utf-8') as dump_file:
        json.dump(jobj, dump_file, ensure_ascii=False, indent=4)
        dump_file.write('\n')


def add_tagged_adoc_str(tag, content):
    tagged_str = str('# tag::{tag}[]\n'
                   '{cont}'
                   '# end::{tag}[]\n\n').format(cont=content, tag=tag)
    return tagged_str


def add_image_to_adoc(image_title, file, tag):
    adoc_str = str('.{title}\n'
                   'image::{file}[{title}]\n').format(title=image_title, file=file)
    adoc_str = add_tagged_adoc_str(tag, adoc_str)
    return adoc_str


def add_inline(content):
    inline_str = str('----\n'
                   '{cont}'
                   '\n----\n').format(cont=content)
    return inline_str


def add_tagged_inline_json(obj, tag):
    adoc_str = '[source, json]\n'
    adoc_str += add_inline(json.dumps(obj, ensure_ascii=False, indent=4))
    adoc_str = add_tagged_adoc_str(tag, adoc_str)
    return adoc_str


def add_layer_to_adoc(cfg):
    return add_tagged_inline_json(get_layer_dump_lst(cfg.prj.model.layers), 'layer')


def add_model_config_to_adoc(cfg):
    model_config_dict = {
        'loss_func': cfg.prj.model.loss_func,
        'optimizer': cfg.prj.model.optimizer,
        'activation': cfg.prj.model.activation,
        'lr': cfg.prj.model.lr,
        'nb_epochs': cfg.prj.train.nb_epochs,
        'batch_size': cfg.prj.train.batch_siz
    }
    return add_tagged_inline_json(model_config_dict, 'config')


def get_res_dict(cfg):
    result_dict = dict()
    result_dict['loss'] = cfg.prj.data.out.metrics[0]
    for i, metric in enumerate(cfg.prj.model.metrics):
        result_dict[metric] = cfg.prj.data.out.metrics[i+1]
    return result_dict


def add_res_to_adoc(cfg):
    result_dict = get_res_dict(cfg)
    return add_tagged_inline_json(result_dict, 'result')


def get_adoc_tag(name):
    return name.split('.')[0]


def add_img_lst_to_adoc(cfg):
    adoc_img_lst = str()
    adoc_str_lst = [('Confusion Matrix', op.get_full_file_nam(cfg, cfg.prj.files.conf_matrix), get_adoc_tag(cfg.prj.files.conf_matrix)),
                    ('Correctly classified samples', op.get_full_file_nam(cfg, cfg.prj.files.correct_image), get_adoc_tag(cfg.prj.files.correct_image)),
                    ('Incorrectly classified samples', op.get_full_file_nam(cfg, cfg.prj.files.incorrect_image), get_adoc_tag(cfg.prj.files.incorrect_image)),
                    ('Metrics', op.get_full_file_nam(cfg, cfg.prj.files.metrics_plot), get_adoc_tag(cfg.prj.files.metrics_plot)),
                    ('Model Summary', op.get_full_file_nam(cfg, cfg.prj.files.model_summary), get_adoc_tag(cfg.prj.files.model_summary)),
                    ('Precision-Recall', op.get_full_file_nam(cfg, cfg.prj.files.prec_recall_curve), get_adoc_tag(cfg.prj.files.prec_recall_curve)),
                    ('ROC', op.get_full_file_nam(cfg, cfg.prj.files.roc_curve), get_adoc_tag(cfg.prj.files.roc_curve)),
                    ]
    for title, file_name, tag in adoc_str_lst:
        adoc_img_lst += add_image_to_adoc(title, file_name, tag)
    return adoc_img_lst


def dump_adoc(cfg):
    adoc_op = add_img_lst_to_adoc(cfg)
    adoc_op += add_layer_to_adoc(cfg)
    adoc_op += add_model_config_to_adoc(cfg)
    adoc_op += add_res_to_adoc(cfg)

    with open(op.get_full_file_nam(cfg, cfg.prj.files.image_lst), 'w') as adoc_fil:
        adoc_fil.write(adoc_op)


def dump(cfg):
    dump_json(cfg)
    dump_adoc(cfg)

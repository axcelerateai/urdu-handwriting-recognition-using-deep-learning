import json
import collections

__all__ = ["get_config"]

def _load_config_file(config_file):
    with open(config_file) as f:
        config = json.load(f)
    return config

def _create_named_tuple(config):
    keys = []
    values = []
    
    for k, v in zip(config.keys(), config.values()):
        keys.append(k)
        values.append(_cast_to_correct_dtype(k,v))

    Config = collections.namedtuple("Config", keys)
    config = Config(*values)

    return config

def _cast_to_correct_dtype(name, value):
    str_types = ["model",
                 "save_dir",
                 "data_folder",
                 "char_or_lig",
                 "optimizer",
                 "rnn_unit_type",
                 "rnn_type",
                 "cnn_activation",
                 "encoder_unit_type",
                 "encoder_type",
                 "attention_type",
                 "decoder_unit_type",
                 "decoder_type",
                 "lm_path"]

    bool_types = ["use_gpu",
                  "debug_mode",
                  "use_dynamic_lengths",
                  "use_teacher_forcing",
                  "use_sparse_labels",
                  "save_best",
                  "restore_previous_model",
                  "restore_best_model",
                  "early_stopping",
                  "verbose",
                  "save_alignments",
                  "flip_image",
                  "anneal_lr",
                  "use_linear_projection",
                  "extract_features",
                  "use_attention",
                  "output_attention",
                  "pass_hidden_state",
                  "do_scheduled_sampling",
                  "anneal_not_sampling_prob"]

    int_types = ["max_outputs",
                 "stop_after_num_epochs",
                 "batch_size",
                 "buckets",
                 "num_epochs",
                 "print_every",
                 "anneal_lr_every",
                 "rnn_num_layers",
                 "rnn_num_residual_layers",
                 "rnn_num_units",
                 "cnn_num_layers",
                 "cnn_num_residual_layers",
                 "encoder_num_layers",
                 "encoder_num_residual_layers",
                 "encoder_num_units",
                 "embed_size",
                 "attention_num_units",
                 "decoder_num_layers",
                 "decoder_num_residual_layers",
                 "decoder_num_units",
                 "beam_width",
                 "ngrams"]

    float_types = ["restore_best_model_range",
                   "lr",
                   "anneal_lr_rate",
                   "l2_regularizer_scale",
                   "initial_not_sampling_prob",
                   "anneal_not_sampling_prob_rate",
                   "anneal_not_sampling_prob_every",
                   "coverage_penalty_weight",
                   "length_penalty_weight",
                   "alpha",
                   "beta",
                   "discard_probability"]

    int_none_types = ["num_train",
                      "image_width_range"]
    
    float_none_types = ["dropout",
                        "max_grad_norm"]

    list_int_none_types = ["image_size",
                           "cnn_num_filters",
                           "cnn_filter_sizes"]

    list_str_types = ["cnn_paddings",
                      "pool_paddings"]

    list_bool_types = ["do_batch_norm",
                       "cnn_residual_layers"]

    nested_list_int_none_types = ["cnn_strides",
                                  "pool_sizes",
                                  "pool_strides"]

    if name in str_types:
        return_val = value

    elif name in bool_types:
        return_val = _cast_to_bool(value)

    elif name in int_types:
        return_val = int(value)

    elif name in float_types:
        return_val = float(value)

    elif name in int_none_types:
        return_val = _cast_int_none(value)
        
    elif name in float_none_types:
        return_val = _cast_float_none(value)
    
    elif name in list_int_none_types:
        return_val = _convert_to_list(value, _cast_int_none, split_seq=', ')

    elif name in list_str_types:
        return_val = _convert_to_list(value, None, split_seq=', ')

    elif name in list_bool_types:
        return_val = _convert_to_list(value, _cast_to_bool, split_seq=', ')

    elif name in nested_list_int_none_types:
        return_val = _convert_nested_list(value, _cast_int_none)

    else:
        raise ValueError("Unknown field in config file: %s" % name)

    return return_val

def _convert_nested_list(value, fn):
    value_trunc = value[1:-1]
    get_lists = _convert_to_list(value_trunc, None, split_seq='), (')
    return_val = []
    for l in get_lists:
        return_val.append(tuple(_convert_to_list(l, fn, split_seq=',')))

    return return_val

def _convert_to_list(value, fn, split_seq=', '):
    new_value = value.split(split_seq)
    return_val = []
    for nv in new_value:
        if fn is None:
            return_val.append(nv)
        else:
            return_val.append(fn(nv))

    return return_val

def _cast_float_none(value):
    if value == "None":
        return_val = None
    else:
        return_val = float(value)

    return return_val

def _cast_int_none(value):
    if value == "None":
        return_val = None
    else:
        return_val = int(value)

    return return_val

def _cast_to_bool(value):
    return value == "True"

def get_config(path):
    config_dict = _load_config_file(path)
    config = _create_named_tuple(config_dict)

    return config

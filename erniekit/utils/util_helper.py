# -*- coding: utf-8 -*
"""import"""
import collections
import json
import unicodedata
from collections import OrderedDict
import six
from ..common.rule import MaxTruncation
from . import params
import os, tarfile


def append_name(name, postfix):
    """ append name with postfix """
    if name is None:
        ret = None
    elif name == '':
        ret = postfix
    else:
        ret = '%s_%s' % (name, postfix)
    return ret


def parse_data_config(config_path):
    """truncate_seq_pair
    :param config_path:
    :return:
    """
    try:
        with open(config_path) as json_file:
            config_dict = json.load(json_file, object_pairs_hook=OrderedDict)
    except Exception:
        raise IOError("Error in parsing Ernie model config file '%s'" % config_path)
    else:
        return config_dict


def parse_version_code(version_str, default_version_code=1.5):
    """
    parser paddle fluid version code to float type
    :param version_str:
    :param default_version_code:
    :return:
    """
    if version_str:
        v1 = version_str.split(".")[0:2]
        v_code_str = ".".join(v1)
        v_code = float(v_code_str)
        return v_code
    else:
        return default_version_code


def truncation_words(words, max_seq_length, truncation_type):
    """
    :param words:
    :param max_seq_length:
    :param truncation_type:
    :return:
    """
    if len(words) > max_seq_length:
        if truncation_type == MaxTruncation.KEEP_HEAD:
            words = words[0: max_seq_length]
        elif truncation_type == MaxTruncation.KEEP_TAIL:
            tmp = words[0: max_seq_length - 1]
            tmp.append(words[-1])
            words = tmp
        elif truncation_type == MaxTruncation.KEEP_BOTH_HEAD_TAIL:
            tmp = words[1: max_seq_length - 2]
            tmp.insert(0, words[0])
            tmp.insert(max_seq_length - 1, words[-1])
            words = tmp
        else:
            words = words[0: max_seq_length]

    return words


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    :param tokens_a:
    :param tokens_a:
    :param max_length:
    :return:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def save_meta_data(data_dict, save_file, mode="add"):
    """
    :param data_dict:
    :param save_file:
    :param mode: 保存模式: override, add
    :return:
    """
    # 目标文件已存在且是追加模式的时候，需要先将原来的dict读出来，再用新的dict去更新原来的dict，最后保存
    if os.path.exists(save_file) and mode == "add":
        meta_dict = params.from_file(save_file)
        _meta = params.replace_none(meta_dict)
        _meta.update(data_dict)
        json_str = json.dumps(_meta)
    else:
        json_str = json.dumps(data_dict)
    with open(save_file, 'w') as json_file:
        json_file.write(json_str)


def get_model_paths(path_checkpoint, path_inference_model, steps, need_encryption=False):
    """ 通过step和trainer_param配置中的output路径，计算出模型存储时需要用到的所有路径
    :param path_checkpoint:
    :param path_inference_model:
    :param steps:
    :param need_encryption:
    :return:
    """
    suffix = ""
    infer_meta_name = "infer_data_params.json"
    model_meta_name = "model.meta"
    if need_encryption:
        suffix = "_enc"

    # 文件保存的原始路径，当不需要加密的时候，原始路径和最终的模型保存路径是同一个
    checkpoint_original_name = "checkpoints_step_" + str(steps)
    checkpoint_original_model_path = os.path.join(path_checkpoint, checkpoint_original_name)
    checkpoint_name = "checkpoints_step_" + str(steps) + suffix
    checkpoint_meta_path = os.path.join(path_checkpoint, checkpoint_name, model_meta_name)
    checkpoint_model_path = os.path.join(path_checkpoint, checkpoint_name)
    checkpoint_infer_meta_path = os.path.join(path_checkpoint, checkpoint_name, infer_meta_name)
    checkpoint_irepo_meta_path = os.path.join(path_checkpoint, checkpoint_name + ".meta")

    inference_original_name = "inference_step_" + str(steps)
    inference_original_model_path = os.path.join(path_inference_model, inference_original_name)
    inference_name = "inference_step_" + str(steps) + suffix
    inference_meta_path = os.path.join(path_inference_model, inference_name, model_meta_name)
    inference_model_path = os.path.join(path_inference_model, inference_name)
    inference_infer_meta_path = os.path.join(path_inference_model, inference_name, infer_meta_name)
    inference_irepo_meta_path = os.path.join(path_inference_model, inference_name + ".meta")

    path_dict = collections.OrderedDict()
    path_dict["checkpoints_name"] = checkpoint_name
    path_dict["checkpoints_original_name"] = checkpoint_original_name
    path_dict["checkpoints_original_model_path"] = checkpoint_original_model_path
    path_dict["checkpoints_model_path"] = checkpoint_model_path
    path_dict["checkpoints_meta_path"] = checkpoint_meta_path
    path_dict["checkpoints_infer_meta_path"] = checkpoint_infer_meta_path
    path_dict["checkpoints_irepo_meta_path"] = checkpoint_irepo_meta_path
    path_dict["inference_name"] = inference_name
    path_dict["inference_original_name"] = inference_original_name
    path_dict["inference_original_model_path"] = inference_original_model_path
    path_dict["inference_model_path"] = inference_model_path
    path_dict["inference_meta_path"] = inference_meta_path
    path_dict["inference_infer_meta_path"] = inference_infer_meta_path
    path_dict["inference_irepo_meta_path"] = inference_irepo_meta_path

    return path_dict


def format_convert_bio(dir_path, vocab_path=None):
    """return"""

    def is_alphabet_or_digit(c):
        """return"""
        alphabet = list(u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digit = list(u"0123456789.")
        if c in alphabet or c in digit:
            return True
        return False

    vocab_map = collections.OrderedDict()
    count = 0
    filelist = os.listdir(dir_path)
    for file_path in filelist:
        if file_path.endswith(".txt"):
            file_path = os.path.join(dir_path, file_path)
            with open(file_path, "r") as fp1:
                with open(file_path + "_bio", "w") as fp2:
                    for line in fp1:
                        try:
                            tokens, triple, offset = line.strip("\n").split("\t")
                            _, _, predicate = triple.split(" ")
                            subject_start, subject_end, object_start, object_end = offset.split(" ")
                        except Exception:
                            print(line.strip("\n"))
                            continue
                        tokens = list(convert_to_unicode(tokens))
                        labels = ["O"] * len(tokens)
                        labels[int(subject_start)] = "B-" + predicate + "@" + "S"
                        for i in range(int(subject_start) + 1, int(subject_end) + 1):
                            labels[i] = "I"
                        if not ("B-" + predicate + "@" + "S") in vocab_map:
                            vocab_map["B-" + predicate + "@" + "S"] = count
                            count += 1
                        labels[int(object_start)] = "B-" + predicate + "@" + "O"
                        for i in range(int(object_start) + 1, int(object_end) + 1):
                            labels[i] = "I"
                        if not ("B-" + predicate + "@" + "O") in vocab_map:
                            vocab_map["B-" + predicate + "@" + "O"] = count
                            count += 1
                        # sub_tokens = []
                        # sub_labels = []
                        # sub_token = ""
                        # sub_label = ""
                        # is_first = True
                        # for i in range(len(tokens)):
                        # if is_alphabet_or_digit(tokens[i]):
                        # sub_token += tokens[i]
                        # if is_first:
                        # sub_label = labels[i]
                        # is_first = False
                        # else:
                        # if sub_token != "":
                        # sub_tokens.append(sub_token)
                        # sub_labels.append(sub_label)
                        # sub_token = ""
                        # is_first = True
                        # sub_tokens.append(tokens[i])
                        # sub_labels.append(labels[i])
                        # if sub_token != "":
                        # sub_tokens.append(sub_token)
                        # sub_labels.append(sub_label)
                        # if len(sub_tokens) != len(sub_labels) or u"" in sub_tokens:
                        # print("Hello", "*****")
                        # continue
                        fp2.write(" ".join(tokens) + "\t")
                        fp2.write(" ".join(labels) + "\n")
            os.remove(file_path)
    vocab_map["I"] = count
    vocab_map["O"] = count + 1
    # if vocab_path:
    #    with open(vocab_path, "w") as fp3:
    #        for key in vocab_map.keys():
    #            fp3.write(key + "\t" + str(vocab_map[key]) + "\n")
    #    return len(vocab_map)
    return None


def make_targz(output_filename, source_dir):
    """压缩某个文件为tar.gz
    :param output_filename: 压缩包路径
    :param source_dir: 待压缩原始路径
    :return:
    """
    errcode = -1
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        errcode = 0
    return errcode


def get_warmup_and_linear_decay(max_steps, warmup_steps):
    """ warmup linear decay function """
    return lambda step: min(step / warmup_steps, 1. - (step - warmup_steps) / (max_steps - warmup_steps))


_work_dir = None


def get_work_path(path):
    """
    get_work_path
    """
    if not path or not _work_dir or path[0] in './':
        return path
    return os.path.join(_work_dir, path)


# paddle _import_module_from_library函数重写，待paddle上线后可废弃
import logging
import sys
from paddle.fluid import core
from paddle.fluid.framework import OpProtoHolder
import threading
import atexit
import textwrap
from importlib import machinery

logger = logging.getLogger("utils.util_helper")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def log_v(info, verbose=True):
    """
    Print log information on stdout.
    """
    if verbose:
        logger.info(info)


OS_NAME = sys.platform
IS_WINDOWS = OS_NAME.startswith('win')


def load_op_meta_info_and_register_op(lib_filename):
    """
    load of meta info and register op
    """
    core.load_op_meta_info_and_register_op(lib_filename)
    return OpProtoHolder.instance().update_op_proto()


def import_module_from_library_wenxin(module_name, build_directory, verbose=False):
    """
    Load shared library and import it as callable python module.
    """
    if IS_WINDOWS:
        dynamic_suffix = '.pyd'
    elif OS_NAME.startswith('darwin'):
        dynamic_suffix = '.dylib'
    else:
        dynamic_suffix = '.so'
    ext_path = os.path.join(build_directory, module_name + dynamic_suffix)
    if not os.path.exists(ext_path):
        raise ValueError("Extension path: {} does not exist.".format(
            ext_path))

    # load custom op_info and kernels from .so shared library
    log_v('loading shared library from: {}'.format(ext_path), verbose)
    op_names = load_op_meta_info_and_register_op(ext_path)

    # generate Python api in ext_path
    return _generate_python_module(module_name, op_names, build_directory,
                                   verbose)


DEFAULT_OP_ATTR_NAMES = [
    core.op_proto_and_checker_maker.kOpRoleAttrName(),
    core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
    core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
    core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(),
    core.op_proto_and_checker_maker.kOpDeviceAttrName(),
    core.op_proto_and_checker_maker.kOpWithQuantAttrName()
]


def parse_op_info(op_name):
    """
    Parse input names and outpus detail information from registered custom op
    from OpInfoMap.
    """
    if op_name not in OpProtoHolder.instance().op_proto_map:
        raise ValueError(
            "Please load {} shared library file firstly by "
            "`paddle.utils.cpp_extension.load_op_meta_info_and_register_op(...)`".
                format(op_name))
    op_proto = OpProtoHolder.instance().get_op_proto(op_name)

    in_names = [x.name for x in op_proto.inputs]
    out_names = [x.name for x in op_proto.outputs]
    attr_names = [
        x.name for x in op_proto.attrs if x.name not in DEFAULT_OP_ATTR_NAMES
    ]

    return in_names, out_names, attr_names


def _get_api_inputs_str(op_name):
    """
    Returns string of api parameters and inputs dict.
    """
    in_names, out_names, attr_names = parse_op_info(op_name)
    # e.g: x, y, z
    param_names = in_names + attr_names
    # NOTE(chenweihang): we add suffix `@VECTOR` for std::vector<Tensor> input,
    # but the string contains `@` cannot used as argument name, so we split
    # input name by `@`, and only use first substr as argument
    params_str = ','.join([p.split("@")[0].lower() for p in param_names])
    # e.g: {'X': x, 'Y': y, 'Z': z}
    ins_str = "{%s}" % ','.join([
        "'{}' : {}".format(in_name, in_name.split("@")[0].lower())
        for in_name in in_names
    ])
    # e.g: {'num': n}
    attrs_str = "{%s}" % ",".join([
        "'{}' : {}".format(attr_name, attr_name.split("@")[0].lower())
        for attr_name in attr_names
    ])
    # e.g: ['Out', 'Index']
    outs_str = "[%s]" % ','.join(["'{}'".format(name) for name in out_names])
    return [params_str, ins_str, attrs_str, outs_str]


def _custom_api_content(op_name):
    (params_str, ins_str, attrs_str, outs_str) = _get_api_inputs_str(op_name)

    API_TEMPLATE = textwrap.dedent("""
        from paddle.fluid.core import VarBase
        from paddle.fluid.framework import in_dygraph_mode, _dygraph_tracer
        from paddle.fluid.layer_helper import LayerHelper
        def {op_name}({inputs}):
            # prepare inputs and outputs
            ins = {ins}
            attrs = {attrs}
            outs = {{}}
            out_names = {out_names}
            # The output variable's dtype use default value 'float32',
            # and the actual dtype of output variable will be inferred in runtime.
            if in_dygraph_mode():
                for out_name in out_names:
                    outs[out_name] = VarBase()
                _dygraph_tracer().trace_op(type="{op_name}", inputs=ins, outputs=outs, attrs=attrs)
            else:
                helper = LayerHelper("{op_name}", **locals())
                for out_name in out_names:
                    outs[out_name] = helper.create_variable(dtype='float32')
                helper.append_op(type="{op_name}", inputs=ins, outputs=outs, attrs=attrs)
            res = [outs[out_name] for out_name in out_names]
            return res[0] if len(res)==1 else res
            """).lstrip()

    # generate python api file
    api_content = API_TEMPLATE.format(
        op_name=op_name,
        inputs=params_str,
        ins=ins_str,
        attrs=attrs_str,
        out_names=outs_str)

    return api_content


def _load_module_from_file(api_file_path, module_name, verbose=False):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise ValueError("File : {} does not exist.".format(
            api_file_path))

    # Unique readable module name to place custom api.
    log_v('import module from file: {}'.format(api_file_path), verbose)
    ext_name = "_paddle_cpp_extension_" + module_name

    # load module with RWLock
    loader = machinery.SourceFileLoader(ext_name, api_file_path)
    module = loader.load_module()

    return module


def _generate_python_module(module_name,
                            op_names,
                            build_directory,
                            verbose=False):
    """
    Automatically generate python file to allow import or load into as module
    """

    def remove_if_exit(filepath):
        """
        remove if file exit
        """
        if os.path.exists(filepath):
            os.remove(filepath)

    # NOTE: Use unique id as suffix to avoid write same file at same time in
    # both multi-thread and multi-process.
    thread_id = str(threading.currentThread().ident)
    api_file = os.path.join(build_directory,
                            module_name + '_' + thread_id + '.py')
    log_v("generate api file: {}".format(api_file), verbose)

    # delete the temp file before exit python process
    atexit.register(lambda: remove_if_exit(api_file))

    # write into .py file with RWLock
    api_content = [_custom_api_content(op_name) for op_name in op_names]
    with open(api_file, 'w') as f:
        f.write('\n\n'.join(api_content))

    # load module
    custom_module = _load_module_from_file(api_file, module_name, verbose)
    return custom_module

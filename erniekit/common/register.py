# -*- coding: utf-8 -*
"""
:py:class:`Register` 将需要的模块加到注册表中，然后import
"""
import importlib
import traceback

import logging
import os


class Register(object):
    """Register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            """decorator"""
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class RegisterSet(object):
    """RegisterSet"""
    field_reader = Register("field_reader")
    data_set_reader = Register("data_set_reader")
    models = Register("models")
    tokenizer = Register("tokenizer")
    trainer = Register("trainer")
    embedding = Register("embedding")
    inference = Register("inference")

    package_names = ['erniekit.data.field_reader', 'erniekit.data.data_set_reader', 'erniekit.model',
                    'erniekit.data.tokenizer', 'erniekit.modules.token_embedding']
    ALL_MODULES = []

    for package_name in package_names:
        module_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../" + package_name.replace(".", '/'))
        module_files = []
        for file in os.listdir(module_dir):
            if os.path.isfile(os.path.join(module_dir, file)):
                if file.endswith(".py"):
                    module_files.append(file.replace(".py", ""))
                if file.endswith(".so"):
                    module_files.append(file.replace(".so", ""))
        ALL_MODULES.append((package_name, module_files))


def import_modules_plugin():
    """import ernie 插件版需要的包，结合注册机制用
    """
    PLUGIN_MODULES = [("erniekit.data.tokenizer", "tokenization_wp"), ("erniekit.modules", "ernie_origin")]

    for package_name, file_name in PLUGIN_MODULES:
        import_new_module(package_name, file_name)


def import_modules():
    """import需要的包，结合注册机制用
    :return:
    """
    for base_dir, modules in RegisterSet.ALL_MODULES:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)

            except Exception:
                logging.error("error in import modules")
                logging.error("traceback.format_exc():\n%s" % traceback.format_exc())


def import_new_module(package_name, file_name):
    """import一个新的类
    :param package_name: 包名
    :param file_name: 文件名，不需要文件后缀
    :return:
    """
    try:
        if package_name != "":
            full_name = package_name + "." + file_name
        else:
            full_name = file_name
        importlib.import_module(full_name)

    except Exception:
        logging.error("error in import %s" % file_name)
        logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

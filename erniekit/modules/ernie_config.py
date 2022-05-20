# -*- coding: utf-8 -*
"""
ERNIE 网络需要的配置解析
"""
import logging
import six
import json


class ErnieConfig(object):
    """parse ernie config"""

    def __init__(self, config_path):
        """
        :param config_path:
        """
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        """
        :param config_path:
        :return:
        """
        try:
            with open(config_path, 'r') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        """
        :param key:
        :return:
        """
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        """
        :param key, value:
        """
        self._config_dict[key] = value

    def has(self, key):
        """
        :param key:
        :return:
        """
        if key in self._config_dict:
            return True
        return False

    def get(self, key, default_value):
        """
        :param key,default_value:
        :retrun:
        """
        if key in self._config_dict:
            return self._config_dict[key]
        else:
            return default_value

    def print_config(self):
        """
        :return:
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')

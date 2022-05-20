# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了一个hello world的demo类。

Authors: sunyu02(sunyu02@baidu.com)
Date:    2021/05/11 16:14:24
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


class Hello(object):
    """hello world类"""
    def run(self, name="World", *args):
        """主入口方法。

        根据百度python编码规范，注释应当使用google风格。
        可以使用sphinx配合napoleon扩展插件自动生成文档。

        Args:
            name: 名称

        Returns:
            int类型，执行结果，0表示成功

        Raises:
            ValueError: 参数name的取值不合法
        """
        if not name:
            raise ValueError(name)

        print("Hello {0}!".format(name))
        return 0


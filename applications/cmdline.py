# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件提供了命令行工具的入口逻辑。

Authors: sunyu02(sunyu02@baidu.com)
Date:    2021/05/11 16:14:24
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


__all__ = [
    'main',
]


def main(args=None):
    """主程序入口"""
    from . import demo
    if args is None:
        # 如果未传入命令行参数，则直接从sys中读取，并过滤掉第0位的入口文件名
        import sys
        args = sys.argv[1:]

    hello = demo.Hello()
    return hello.run(*args)


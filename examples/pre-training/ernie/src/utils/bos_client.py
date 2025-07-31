# -*- coding: utf-8 -*-
# !/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import time
from pathlib import Path
from collections import namedtuple
from omegaconf import OmegaConf

from baidubce import compat
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient as BceBosClient
from baidubce.exception import BceError
from bce_bns_proxy.proxy import BceBNSProxy

strict_bos_path_pattern = "bos://([^/]+)/(.+)"
loose_bos_path_pattern = "bos://([^/]+)(?:/(.+)?)?"
ParsedBosPath = namedtuple("ParsedBosPath", ["bucket_name", "object_key", "bos_path"])

logger = logging.getLogger(__name__)

DEFAULT_BOS_CLIENT_CONFIF_PATH = os.path.join(
    os.path.dirname(__file__), "bos_client.yaml"
)

__all__ = ["BosClient"]


class BosClient(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    BOS_CLINET_CONFIG_PATH = Path(
        os.environ.get("BOS_CLIENT_CONFIG", DEFAULT_BOS_CLIENT_CONFIF_PATH)
    ).expanduser()

    def __init__(self):
        """
            初始化函数，初始化bos客户端列表和桶名匹配模式列表。
        如果配置了bns，则使用bns进行初始化；否则使用endpoint进行初始化。

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: 如果没有配置bns或endpoint中的任何一个，会引发RuntimeError异常。
        """
        self.bos_client_list = []
        self.bucket_patterns_list = []

        config = OmegaConf.load(self.BOS_CLINET_CONFIG_PATH)
        for local_config in config.local_configs:
            kwargs = OmegaConf.to_object(config.global_config)
            kwargs.update(OmegaConf.to_object(local_config))
            ak = kwargs.pop("ak")
            sk = kwargs.pop("sk")
            endpoint = kwargs.pop("endpoint", None)
            bns = kwargs.pop("bns", None)
            if endpoint is not None:
                endpoint = compat.convert_to_bytes(endpoint)
            if bns is None and endpoint is None:
                raise RuntimeError("至少要配置bns或endpoint中的一个")
            bucket_patterns = kwargs.pop("bucket_patterns")

            if bns is not None:
                # 优先使用bns配置
                bce_config = BceBNSProxy(
                    credentials=BceCredentials(ak, sk),
                    bns=bns,
                    endpoint=endpoint,
                    bns_cache_time=60,  # bns 缓存时间，默认60s
                    bns_retry_time=1,  # bns 重试次数，默认1次
                    bns_retry_gap=0.2,  # bns 重试间隔，默认0.2s
                )
                logger.info(f"使用bns配置初始化bos客户端，bns-[{bns}-ak: {ak}]")
            else:
                bce_config = BceBNSProxy(
                    credentials=BceCredentials(ak, sk),
                    endpoint=endpoint,
                    bns_cache_time=60,  # bns 缓存时间，默认60s
                    bns_retry_time=1,  # bns 重试次数，默认1次
                    bns_retry_gap=0.2,  # bns 重试间隔，默认0.2s
                )
                logger.info(
                    f"使用endpoint配置初始化bos客户端，endpoint-[{endpoint}-ak: {ak}]"
                )
            bos_client = BceBosClient(bce_config)
            self.bos_client_list.append(bos_client)
            self.bucket_patterns_list.append(bucket_patterns)
        self.lookup_table = dict()

    @classmethod
    def parse_bos_path(cls, bos_path, strict=True, raise_error=False):
        """_summary_

        Args:
            bos_path (_type_): _description_
            strict (bool, optional): _description_. Defaults to True.
            raise_error (bool, optional): _description_. Defaults to False.

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        pattern = strict_bos_path_pattern if strict else loose_bos_path_pattern
        match = re.fullmatch(pattern, bos_path)
        if match is None:
            if raise_error:
                raise RuntimeError(f"bos路径[{bos_path}]格式不正确")
            else:
                return None
        bucket_name, object_key = match.groups()
        return ParsedBosPath(bucket_name, object_key, bos_path)

    def lookup_bos_client(self, bucket_name) -> BceBosClient:
        """
            根据bucket_name查找对应的BceBosClient，如果不存在则返回None。
        如果存在多个bucket_pattern匹配该bucket_name，则使用第一个匹配的bucket_pattern。

        Args:
            bucket_name (str): 需要查找的bucket名称。

        Returns:
            BceBosClient, optional: 返回对应的BceBosClient实例，如果不存在则返回None。默认值为None。

        Raises:
            RuntimeError: 当bucket_name没有匹配到任何bucket_pattern时，会抛出RuntimeError异常，提示配置错误。
        """
        index = self.lookup_table.get(bucket_name, None)
        if index is not None:
            return self.bos_client_list[index]
        for index, bucket_patterns in enumerate(self.bucket_patterns_list):
            for bucket_pattern in bucket_patterns:
                if re.fullmatch(bucket_pattern, bucket_name):
                    self.lookup_table[bucket_name] = index
                    return self.bos_client_list[index]
        raise RuntimeError(
            f"bucket_name [{bucket_name}]没有匹配到任何bucket_pattern，请检查配置是否正确"
        )

    def exists(self, bos_path):
        """
            判断BOS路径是否存在，如果不存在则抛出异常。
        如果存在返回True，否则返回False。

        Args:
            bos_path (str): BOS路径，格式为"<bucket-name>/<object-key>"。

        Returns:
            bool: 如果BOS路径存在，则返回True；否则返回False。

        Raises:
            BceError: 当BOS路径不存在时，会抛出BceError异常。
        """
        parsed = self.parse_bos_path(bos_path, raise_error=True)
        bos_client = self.lookup_bos_client(parsed.bucket_name)
        try:
            bos_client.get_object_meta_data(parsed.bucket_name, parsed.object_key)
            return True
        except BceError:
            return False

    def get_bytes(self, bos_path, retry=0, retry_interval=3, get_range=None):
        """
            从BOS获取字节流，支持断点续传和指定范围下载。如果出现网络或服务器错误，会自动重试一定次数。
        如果重试失败，将抛出RuntimeError异常。

        Args:
            bos_path (str): BOS路径，例如"bos://my-bucket/my-object"。
            retry (int, optional): 重试次数，默认为0，表示不进行重试。如果设置为正数，将在出现网络或服务器错误时进行重试。
            retry_interval (int, optional): 每次重试的间隔，单位为秒，默认为3秒。
            get_range (tuple, optional): 指定下载范围，格式为(start, end)，其中start和end都是0-based的索引值，默认为None，表示下载全部内容。

        Returns:
            str: 返回BOS对象的字节流。

        Raises:
            RuntimeError: 当重试次数达到上限仍然无法成功获取字节流时，将抛出此类型的异常。
        """
        if retry < 0:
            raise RuntimeError("参数[retry]必须大于等于0")
        parsed = self.parse_bos_path(bos_path, raise_error=True)
        bos_client = self.lookup_bos_client(parsed.bucket_name)
        for i in range(retry + 1):
            try:
                object_bytes = bos_client.get_object_as_string(
                    parsed.bucket_name, parsed.object_key, range=get_range
                )
                return object_bytes
            except Exception as e:
                if i == retry:
                    logger.warning(
                        f"重试[{retry}]次仍然无法读取[{bos_path}]，详情见报错："
                    )
                    raise e
                else:
                    time.sleep(retry_interval)

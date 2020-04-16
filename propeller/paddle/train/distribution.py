#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import functools
import six
import os
import logging
from time import sleep

import paddle.fluid as F
import paddle.fluid.layers as L

log = logging.getLogger(__name__)
import propeller.util

__all__ = ['init_distribuition_env', 'status']

status = None


class DistributionMode(object):
    LOCAL = 0
    NCCL = 1


class DistributionStatus(object):
    def __init__(self, config):
        if config is None:
            self._mode = DistributionMode.LOCAL
            self._env = None
            self._this = None
        else:
            try:
                self._mode = DistributionMode.NCCL

                cluster = config['cluster']
                task = config['task']['type']
                idx = int(config['task']['index'])
                self._this = cluster[task][idx]

                self._env = cluster['chief'] + cluster.get('worker', [])
                if len(set(self._env)) != len(self._env):
                    raise ValueError('duplicate host in dis_config %s' %
                                     config)

            except KeyError as e:
                raise ValueError(
                    'PROPELLER_DISCONFIG wrong: %s not found in %s' %
                    (e, repr(config)))

    @property
    def mode(self):
        return self._mode

    @property
    def num_replica(self):
        if self._mode == DistributionMode.LOCAL:
            return 1
        elif self._mode == DistributionMode.NCCL:
            return len(self._env)
        else:
            raise ValueError('Got unknow distribution mode %s' %
                             repr(self._mode))

    @property
    def replica_id(self):
        if self._mode == DistributionMode.LOCAL:
            return 0
        elif self._mode == DistributionMode.NCCL:
            return self._env.index(self._this)
        else:
            raise ValueError('Got unknow distribution mode %s' %
                             repr(self._mode))

    @property
    def is_master(self):
        if self._mode == DistributionMode.LOCAL:
            return True
        elif self._mode == DistributionMode.NCCL:
            return self.replica_id == 0
        else:
            raise ValueError('got unknow distribution mode %s' %
                             repr(self._mode))


def _get_paddlestype_disconfig():
    env = os.environ.copy()
    if not ('PADDLE_TRAINER_ID' in env and 'PADDLE_CURRENT_ENDPOINT' in env and
            'PADDLE_TRAINERS_NUM' in env and
            'PADDLE_TRAINER_ENDPOINTS' in env):
        return None
    else:
        ip_port_list = env['PADDLE_TRAINER_ENDPOINTS'].split(',')
        assert len(ip_port_list) == int(env['PADDLE_TRAINERS_NUM'])
        ip_port_self = env['PADDLE_CURRENT_ENDPOINT']
        world = {"chief": [ip_port_list[0]]}
        for ip_port in ip_port_list[1:]:
            world.setdefault('worker', []).append(ip_port)
        self_index = ip_port_list.index(ip_port_self)
        self_type = 'chief' if self_index == 0 else 'worker'
        if self_type == 'worker':
            self_index -= 1
        env_dict = {
            'cluster': world,
            'task': {
                'type': self_type,
                'index': self_index
            }
        }
        return env_dict


dis_config = propeller.util._get_dict_from_environ_or_json_or_file(
    None, 'PROPELLER_DISCONFIG')
if dis_config is None:
    log.debug('no PROPELLER_DISCONFIG found, try paddlestype setting')
    dis_config = _get_paddlestype_disconfig()
    if dis_config is None:
        log.debug('no paddle stype setting found')
status = DistributionStatus(dis_config)


def run_on_master(func):
    """skip function in distribution env"""

    @functools.wraps(func)
    def f(*arg, **kwargs):
        """f"""
        if status is None:
            raise ValueError('distribution mode unkown at this point')
        if status.mode == DistributionMode.LOCAL:
            r = func(*arg, **kwargs)
        elif status.mode == DistributionMode.NCCL:
            if status.is_master:
                r = func(*arg, **kwargs)
            else:
                r = 0  # skip function
        #MPI.COMM_WORLD.Barrier()
        return r

    return f


def init_distribuition_env(program):
    if status.mode == DistributionMode.LOCAL:
        log.info('Initializing local training')
    elif status.mode == DistributionMode.NCCL:
        config = F.DistributeTranspilerConfig()
        config.mode = "nccl2"
        config.nccl_comm_num = 1
        F.DistributeTranspiler(config=config).transpile(
            status.replica_id,
            trainers=','.join(status._env),
            current_endpoint=status._this,
            program=program.train_program,
            startup_program=program.startup_program)
        log.info('Initializing distribution training with config %s' %
                 (repr(dis_config)))
        if status.is_master:
            sleep(30)

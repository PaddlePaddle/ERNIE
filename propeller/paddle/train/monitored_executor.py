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
"""
doc
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import json
from functools import reduce
import six
#from time import time
import time
import shutil
import tarfile
import tempfile

import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from propeller import util
from propeller.types import StopException, ProgramPair, WarmStartSetting, TextoneWarmStartSetting
from propeller.paddle.train import hooks
from . import distribution

log = logging.getLogger(__name__)

__all__ = ['MonitoredExecutor', 'Saver']


def _get_one_place():
    return F.cuda_places()[0] if F.core.is_compiled_with_cuda(
    ) else F.cpu_places()[0]


class RunState(object):
    """serializable Run state object"""

    @classmethod
    def from_dict(cls, d):
        d['step'] = 0
        r = RunState()
        r.__dict__ = d
        return r

    @classmethod
    def from_str(cls, s):
        """doc"""
        j = json.loads(s)
        return cls.from_dict(j)

    def __init__(self):
        """doc"""
        self.__dict__ = {'gstep': 0, 'step': 0, 'time': time.time()}

    @property
    def gstep(self):
        """doc"""
        return self.__dict__.get(
            'gstep',
            self.__dict__.get('global_step'))  # backward compatibility

    @property
    def step(self):
        """doc"""
        return self.__dict__['step']

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __getitem__(self, k):
        return self.__dict__.get(k, None)

    @property
    def time(self):
        """doc"""
        return self.__dict__['time']

    def state_dict(self):
        return self.__dict__

    def __repr__(self):
        """doc"""
        return repr(self.state_dict())

    def serialize(self):
        """doc"""
        return json.dumps(self.state_dict())

    def next(self):
        """doc"""
        newd = dict(
            self.__dict__,
            gstep=self.gstep + 1,
            step=self.step + 1,
            time=time.time())
        ret = RunState()
        ret.__dict__ = newd
        return ret


class Saver(object):
    """checkpoint saver and manager"""

    def __init__(self,
                 save_dir,
                 exe,
                 program,
                 save_prefix='model',
                 max_ckpt_to_keep=None,
                 save_tarckpt=False):
        """doc"""
        self.save_tarckpt = save_tarckpt
        assert isinstance(
            exe, F.Executor
        ), 'expect normal executor to save, got executor of type %s' % repr(
            type(exe))
        self._exe = exe
        self._program = program
        self._save_dir = save_dir
        self._save_prefix = save_prefix
        self._max_ckpt_to_keep = 10 if max_ckpt_to_keep is None else max_ckpt_to_keep

        self.ckpt_info_path = os.path.join(save_dir, 'ckpt_info')

        if os.path.exists(self.ckpt_info_path):
            self.ckpt_list = [
                p.strip() for p in open(self.ckpt_info_path).readlines()
            ]
            log.debug('ckpt_list in this Saver: %s' % (self.ckpt_list))
        else:
            self.ckpt_list = []

    @property
    def last_ckpt(self):
        """doc"""
        return self.ckpt_list[-1] if len(self.ckpt_list) else None

    def _save_program(self, dir):
        F.io.save_persistables(self._exe, dir, self._program.train_program)

    def _load_program(self, dir, predicate_fn=None):
        if predicate_fn is None:

            def _fn(v):
                vpath = os.path.join(dir, v.name)
                if F.io.is_persistable(v):
                    if os.path.exists(vpath):
                        return True
                    else:
                        log.warning('var %s not found in checkpoint, ignored' %
                                    v.name)
                return False

            predicate_fn = _fn
        try:
            F.io.load_vars(
                self._exe,
                dir,
                main_program=self._program.train_program,
                predicate=predicate_fn)
        except F.core.EnforceNotMet as e:
            log.exception(e)
            raise RuntimeError(
                'can not load model from %s, is this a textone checkpoint?' %
                dir)

    def tarball(self, src_dir, output_name):
        with tarfile.open(output_name, "w:") as tar:
            tar.add(src_dir, arcname=os.path.basename(src_dir))

    def save(self, state):
        """doc"""
        save_name = '%s_%d' % (self._save_prefix, state.gstep)
        save_dir = os.path.join(self._save_dir, save_name)
        tmp_dir = os.path.join(self._save_dir, 'tmp')
        try:
            shutil.rmtree(save_dir)
            shutil.rmtree(tmp_dir)
        except OSError:
            pass
        log.debug('saving step %d to %s' % (state.gstep, save_dir))
        self._save_program(tmp_dir)

        shutil.move(tmp_dir, save_dir)
        meta = state.serialize()
        open(os.path.join(save_dir, 'meta'), 'w').write(meta)

        if self.save_tarckpt:
            now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            save_dir_tar = save_dir + '_' + now + '.tar'
            tar_name = os.path.basename(save_dir_tar)
            log.debug('taring %s to %s' % (save_dir, save_dir_tar))
            self.tarball(save_dir, save_dir_tar)
            shutil.rmtree(save_dir)
            save_name = tar_name

        self.ckpt_list.append(save_name)
        if len(self.ckpt_list) > self._max_ckpt_to_keep:
            ckpt_to_keep = self.ckpt_list[-self._max_ckpt_to_keep:]
            ckpt_to_remove = set(self.ckpt_list) - set(ckpt_to_keep)
            self.ckpt_list = ckpt_to_keep
            for ckpt in ckpt_to_remove:
                ckpt_dir = os.path.join(self._save_dir, ckpt)
                if os.path.exists(ckpt_dir):
                    rm = shutil.rmtree if os.path.isdir(
                        ckpt_dir) else os.remove
                    rm(ckpt_dir)
                    log.debug('No. of ckpt exceed %d, clean up: %s' %
                              (self._max_ckpt_to_keep, ckpt_dir))
        open(self.ckpt_info_path, 'w').write('\n'.join(self.ckpt_list))

    def restore(self, ckpt=-1):
        """doc"""
        if isinstance(ckpt, int):
            try:
                path = os.path.join(self._save_dir, self.ckpt_list[ckpt])
            except IndexError:
                raise ValueError('invalid restore ckpt number %d' % ckpt)
        elif isinstance(ckpt, six.string_types):
            if not os.path.exists(ckpt):
                raise ValueError('ckpt: %s not found' % ckpt)
            path = ckpt
        else:
            raise ValueError('ckpt type not understood %s' % repr(ckpt))

        if os.path.isfile(path) and tarfile.is_tarfile(path):
            log.info('restore from tar file : {}'.format(path))
            tf = tarfile.open(path)
            dirs = [m for m in tf.getmembers() if m.isdir()]
            assert (len(dirs) == 1), dirs
            tmp_dir = tempfile.mkdtemp()
            log.info('extracting to : {}'.format(tmp_dir))
            tf.extractall(tmp_dir)
            path = os.path.join(tmp_dir, dirs[0].name)
            log.info('model path : {}'.format(path))

        meta_file = os.path.join(path, 'meta')
        if not os.path.exists(meta_file):
            raise RuntimeError('meta not found in restore dir: %s' % path)
        state = RunState.from_str(open(meta_file).read())
        log.info('restore from ckpt %s, ckpt-status: %s' % (path, repr(state)))

        self._load_program(path)
        return state


class SaverV2(Saver):
    def _save_program(self, dir):
        save_path = os.path.join(dir, 'ckpt')
        F.save(self._program.train_program, save_path)

    def _load_program(self, dir, predicate_fn=None):
        save_path = os.path.join(dir, 'ckpt')
        if not os.path.exists(save_path + '.pdparams'):
            try:
                log.warn('failed to load model, try old-styled saver')
                super(SaverV2, self)._load_program(
                    dir, predicate_fn=predicate_fn)
            except F.core.EnforceNotMet as e:
                log.exception(e)
                raise RuntimeError(
                    'can not load model from %s, is this a textone checkpoint?'
                    % dir)
        else:
            sd = F.load_program_state(save_path)
            F.set_program_state(self._program.train_program, sd)


TextoneTrainer = None


class MonitoredExecutor(object):
    """An Executor wrapper handling the train loop"""
    saver_class = SaverV2  # will change if textone enabled

    def __init__(
            self,
            executor,
            program,
            loss=None,  #must set in train
            state=None,
            run_config=None,  #none if not load
            run_hooks=[],
            warm_start_setting=None, ):
        if not isinstance(executor, F.Executor):
            raise ValueError('PE is no longer supported')
        if isinstance(executor, F.ParallelExecutor):
            raise ValueError('ParallelExecutor is deprecatd, use Executor')
        if not isinstance(program, ProgramPair):
            raise ValueError('Expect ProgramPair, got %r' % type(program))
        self._exe = executor
        self._hooks = run_hooks
        self._state = RunState()  # might be overwrite in freeze
        self._program = program
        self._loss = loss
        self._warm_start_setting = warm_start_setting
        self._saver = None  # will set in prepare
        self.result = None  # will set after train
        if run_config is not None:
            self._model_dir = run_config.model_dir
            self._save_dir = run_config.model_dir
            self._save_steps = run_config.save_steps
            self._skip_steps = run_config.skip_steps if run_config.skip_steps else 100
            self._save_prefix = 'model'
            self._max_ckpt = run_config.max_ckpt
            self._save_tarckpt = False
            if hasattr(run_config,
                       'save_tarckpt') and run_config.save_tarckpt is True:
                self._save_tarckpt = True

    @property
    def state(self):
        """doc"""
        return self._state

    def init_or_restore_variables(self, ckpt=-1):
        """
        init vars or restore vars from model_dir
        call before train
        """
        # The order of this 2 steps really matters
        # 1. init train

        F.Executor(_get_one_place()).run(self._program.startup_program)
        # 2. restore param

        self._saver = self.saver_class(
            self._model_dir,
            F.Executor(_get_one_place()),
            program=self._program,
            max_ckpt_to_keep=self._max_ckpt,
            save_tarckpt=self._save_tarckpt)

        if self._warm_start_setting is not None:
            if not os.path.exists(self._warm_start_setting.from_dir):
                raise ValueError('warm start dir not exists: %s' %
                                 self._warm_start_setting.from_dir)

            if isinstance(self._warm_start_setting, WarmStartSetting):
                log.info("warm start from %s" %
                         self._warm_start_setting.from_dir)
                if (not type(self._saver) is Saver) and (
                        not type(self._saver) is SaverV2):
                    raise ValueError(
                        'try to warm start from standart dir, but textone enabled'
                    )
                if self._warm_start_setting.predicate_fn is not None:

                    def _fn(v):
                        ret = self._warm_start_setting.predicate_fn(v)
                        if ret:
                            log.info('warm start: %s' % v.name)
                        return ret

                    self._saver._load_program(
                        self._warm_start_setting.from_dir, predicate_fn=_fn)
                else:
                    raise NotImplementedError()
            elif isinstance(self._warm_start_setting, TextoneWarmStartSetting):
                if not type(self._saver) is TextoneTrainer:
                    raise ValueError(
                        'try to warm start from textone pretrain dir, but textone not enabled'
                    )
                log.info("[texone] warm start from %s" %
                         self._warm_start_setting.from_dir)
                self._saver._load_pretrained(self._warm_start_setting.from_dir)
            else:
                raise ValueError(
                    'expect _warm_start_setting to be TextoneWarmStartSetting of WarmStartSetting, got %s'
                    % repr(self._warm_start_setting))

        if self._saver.last_ckpt is not None:
            self._state = self._saver.restore(ckpt)

    def _freeze(self):
        """
        call before enter train loop
        convert program to compiled program
        will do nothing if loss is None i.e. not in train mode
        """
        if self._loss is None:
            #log.debug('will not freeze a program without loss')
            return
        if isinstance(self._program.train_program, F.compiler.CompiledProgram):
            #log.debug('program has already been built')
            return
        exec_strategy = F.ExecutionStrategy()
        exec_strategy.num_threads = 4  #2 for fp32 4 for fp16
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_iteration_per_drop_scope = 10  #important shit

        build_strategy = F.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False
        build_strategy.enable_sequential_execution = True  # prevent hang
        #build_strategy.fuse_broadcast_ops = True
        build_strategy.num_trainers = distribution.status.num_replica
        build_strategy.trainer_id = distribution.status.replica_id
        build_strategy.memory_optimize = True

        log.info('replica id %d of %d' % (distribution.status.replica_id,
                                          distribution.status.num_replica))

        program = F.CompiledProgram(
            self._program.train_program).with_data_parallel(
                loss_name=self._loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        self._program = ProgramPair(
            train_program=program,
            startup_program=self._program.startup_program)

    def __enter__(self):
        """
        prepapre before enter train loop
        """
        if F.core.is_compiled_with_cuda():
            log.info('propeller runs in CUDA mode')
        else:
            log.info('propeller runs in CPU mode')

        #log.debug('freezing program')
        self._freeze()
        #log.debug('done freezing')
        log.info('********** Start Loop ************')
        # TODO init

        self.result = None
        for h in self._hooks:
            log.debug('train loop has hook %s' % h)
            h.before_train(self._program, self._state)
        return self

    def run(self, fetch_list=[], *args, **kwargs):
        """
        wrapper for Executor.run
        """
        #log.debug('Executor running step %d' % self._state.gstep)
        if self._hooks:
            fetch_list = [fetch_list]
            for h in self._hooks:
                #log.debug('calling hook.before_run %s' % h)
                fetch = h.before_run(self._state)
                fetch_list.append(fetch)
            fetch_list_len = map(len, fetch_list)
            fetch_list, schema = util.flatten(fetch_list)
            fetch_list = [
                f.name if not isinstance(f, six.string_types) else f
                for f in fetch_list
            ]
            #if len(set(fetch_list)) != len(fetch_list):
            #    log.error('strange shit happend when fetch list has idetity tensors %s' % fetch_list)
            #log.debug(fetch_list)
            res = self._exe.run(self._program.train_program,
                                fetch_list=fetch_list,
                                *args,
                                **kwargs)
            res = [self._merge_result(r) for r in res]
            #log.debug(res)

            res = util.unflatten(res, schema)
            ret, res = res[0], res[1:]
            for r, h in zip(res, self._hooks):
                #log.debug('calling hook.after_run')
                h.after_run(r, self._state)

            if any(map(lambda i: i.should_stop(self._state), self._hooks)):
                raise StopException('hook call stop')
        else:
            ret = self._exe.run(self._program.train_program,
                                fetch_list=fetch_list,
                                *args,
                                **kwargs)
        self._state = self._state.next()
        return ret

    def __exit__(self, err_type, err_value, trace):
        """
        clean up things and report hook result when exit train loop
        """
        if (err_type is None) or isinstance(err_value, (
                F.core.EOFException, StopException, KeyboardInterrupt)):
            try:
                log.info('********** Stop Loop ************')
                self.result = []
                for h in self._hooks:
                    self.result.append(
                        h.after_train(self._program, self._state))
            except Exception as e:
                log.exception('error occur after loop %s' % repr(e))
        else:
            log.info('********** Interupt Loop ************')
            log.exception('error occur during loop %s: %s' %
                          (err_type, err_value))

    def _merge_result(self, ls):
        """
        merge results from multi gpu cards
        """
        dev_count = len(self._program.train_program._places) if isinstance(
            self._program.train_program, F.compiler.CompiledProgram) else 1
        if dev_count == 1:
            return ls
        else:
            shape = (-1, ls.shape[0] // dev_count) + ls.shape[1:]
            ret = np.reshape(ls, shape).mean(axis=0)
            return ret

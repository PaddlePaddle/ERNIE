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
"""train hooks"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import six
import os
import itertools

import numpy as np
import logging
import paddle.fluid as F
import paddle.fluid.layers as L

from propeller import util
from propeller.paddle.train import distribution
from propeller.paddle.train.metrics import Metrics

__all__ = [
    'RunHook', 'TqdmProgressBarHook', 'TqdmNotebookProgressBarHook',
    'CheckpointSaverHook', 'LoggingHook', 'StopAtStepHook', 'EvalHook'
]

log = logging.getLogger(__name__)


class RunHook(object):
    """RunHook Base class"""

    def __init__(self):
        """doc"""
        pass

    def before_train(self, program):
        """doc"""
        pass

    def before_run(self, state):
        """doc"""
        return []

    def after_run(self, res_list, state):
        """doc"""
        pass

    def should_stop(self, state):
        """doc"""
        return False

    def after_train(self):
        """doc"""
        pass


class TqdmProgressBarHook(RunHook):
    """show a progress bar when training"""

    def __init__(self, max_steps, desc=None):
        """doc"""
        self.tqdm = None
        import tqdm
        from propeller import log as main_log
        hdl = main_log.handlers[0]

        class _TqdmLogginHandler(logging.Handler):
            def emit(self, record):
                """doc"""
                try:
                    msg = self.format(record)
                    tqdm.tqdm.write(msg, file=sys.stderr)
                    self.flush()
                except (KeyboardInterrupt, SystemExit) as e:
                    raise e
                except:
                    self.handleError(record)

        tqdm_hdl = _TqdmLogginHandler()
        tqdm_hdl.setFormatter(hdl.formatter)
        main_log.removeHandler(hdl)
        main_log.addHandler(tqdm_hdl)
        self.tqdm = tqdm.tqdm(total=max_steps, desc=None)

    def before_run(self, state):
        self.tqdm.n = state.gstep
        return []

    def __del__(self):
        if self.tqdm:
            self.tqdm.close()


class TqdmNotebookProgressBarHook(RunHook):
    """show a progress bar when training"""

    def __init__(self, max_steps, desc=None):
        """doc"""
        self.tqdm = None
        import tqdm
        from propeller import log as main_log
        hdl = main_log.handlers[0]

        class _TqdmLogginHandler(logging.Handler):
            def emit(self, record):
                """doc"""
                try:
                    msg = self.format(record)
                    tqdm.tqdm.write(msg, file=sys.stderr)
                    self.flush()
                except (KeyboardInterrupt, SystemExit) as e:
                    raise e
                except:
                    self.handleError(record)

        tqdm_hdl = _TqdmLogginHandler()
        tqdm_hdl.setFormatter(hdl.formatter)
        main_log.removeHandler(hdl)
        main_log.addHandler(tqdm_hdl)
        self.tqdm = tqdm.tqdm_notebook(total=max_steps, desc=None)

    def before_run(self, state):
        """doc"""
        self.tqdm.n = state.gstep
        self.tqdm.refresh()
        return []

    def __del__(self):
        """doc"""
        if self.tqdm:
            self.tqdm.close()


class LoggingHook(RunHook):
    """log tensor in to screan and tensorboard"""

    def __init__(self,
                 loss,
                 per_step=10,
                 skip_step=100,
                 summary_writer=None,
                 summary_record=None):
        """doc"""
        if per_step is None or skip_step is None:
            raise ValueError('wrong step argument, per step: %d skip_step %d' %
                             (per_step, skip_step))
        self.loss = loss
        self.per_step = per_step
        self.skip_step = skip_step
        self.summary_record = summary_record
        self.writer = summary_writer
        self.last_state = None

    def before_train(self, program):
        """doc"""
        if self.summary_record:
            if self.summary_record.scalar:
                self.s_name, self.s_tolog = zip(*self.summary_record.scalar)
            else:
                self.s_name, self.s_tolog = [], []

            if self.summary_record.histogram:
                self.h_name, self.h_tolog = zip(*self.summary_record.histogram)
            else:
                self.h_name, self.h_tolog = [], []

    def before_run(self, state):
        """doc"""
        if state.gstep % self.per_step == 0 and state.step > self.skip_step:
            ret = [self.loss]
            if self.summary_record:
                ret += self.s_tolog
                ret += self.h_tolog
            return ret
        else:
            return []

    def after_run(self, res_list, state):
        """doc"""
        if state.gstep % self.per_step == 0 and state.step > self.skip_step:
            if not self.summary_record:
                return

            loss = float(res_list[0])
            s_np = res_list[1:1 + len(self.s_name)]
            h_np = res_list[1 + len(self.s_name):1 + len(self.s_name) + len(
                self.h_name)]

            if self.last_state is not None:
                speed = (state.gstep - self.last_state.gstep) / (
                    state.time - self.last_state.time)
            else:
                speed = -1.
            self.last_state = state

            # log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('loss', loss, state.gstep)
                for name, t in zip(self.s_name, s_np):
                    if np.isnan(t).any():
                        log.warning('Nan summary: %s, skip' % name)
                    else:
                        self.writer.add_scalar(name, t, state.gstep)

                for name, t in zip(self.h_name, h_np):
                    if np.isnan(t).any():
                        log.warning('Nan summary: %s, skip' % name)
                    else:
                        self.writer.add_histogram(name, t, state.gstep)

                if speed > 0.:
                    self.writer.add_scalar('global_step', speed, state.gstep)

            # log to stdout
            log.debug('\t'.join([
                'step: %d' % state.gstep,
                'steps/sec: %.5f' % speed,
                'loss: %.5f' % loss,
                '' if self.summary_record is None else ' '.join(
                    map(lambda t: '%s:%s' % t, zip(self.s_name, s_np))),
            ]))


class StopAtStepHook(RunHook):
    """stop training at some step"""

    def __init__(self, stop_global_step, stop_step):
        """doc"""
        self._stop_gstep = stop_global_step
        self._stop_step = stop_step

    def should_stop(self, state):
        """doc"""
        if (self._stop_gstep and state.gstep >= self._stop_gstep) or \
           (self._stop_step and state.step >= self._stop_step):
            log.info('StopAtStepHook called stop')
            return True
        else:
            return False


class EvalHook(RunHook):
    """hook this on a eval Executor"""

    def __init__(self, metrics, summary_writer=None):
        """doc"""
        self.writer = summary_writer
        self._result = None

        if not isinstance(metrics, dict):
            raise ValueError('metrics should be dict, got %s' % repr(metrics))

        for k, m in six.iteritems(metrics):
            if not isinstance(m, Metrics):
                raise ValueError(
                    'metrics %s should be instance of propeller.Metrics, got %s'
                    % (k, repr(m)))

        if len(metrics):
            self.names = list(metrics.keys())
            self.metrics = list(metrics.values())
        else:
            self.names, self.metrics = [], []

    def before_train(self, program):
        """doc"""
        for m in self.metrics:
            m.reset()

    def before_run(self, state):
        """doc"""
        ls = [m.tensor for m in self.metrics]
        for i in ls:
            if not (isinstance(i, list) or isinstance(i, tuple)):
                raise ValueError(
                    'metrics should return tuple or list of tensors, got %s' %
                    repr(i))
            for ii in i:
                if not isinstance(ii, F.framework.Variable):
                    raise ValueError(
                        'metrics tensor be propeller.train.Metrics, got %s of type %s'
                        % (repr(ii), type(ii)))
        ls_flt, self.schema = util.flatten(ls)
        #log.debug(ls_flt)
        return ls_flt

    def after_run(self, res_list, state):
        """doc"""
        res = util.unflatten(res_list, self.schema)
        for r, m in zip(res, self.metrics):
            m.update(r)

    @property
    def result(self):
        """doc"""
        return self._result

    def after_train(self):
        """doc"""
        printable = []
        self._result = {}
        for n, m in zip(self.names, self.metrics):
            val = m.eval()
            self._result[n] = val

        return self.result


class CheckpointSaverHook(RunHook):
    """Save checkpoint every n step"""

    def __init__(self, saver, per_step=10, skip_step=100):
        """doc"""
        self.saver = saver
        self.per_step = per_step
        self.skip_step = skip_step

    def after_run(self, res_list, state):
        """doc"""
        if state.gstep % self.per_step == 0 and \
                state.step > self.skip_step:
            self.saver.save(state)

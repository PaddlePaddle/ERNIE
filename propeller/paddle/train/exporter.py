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
exporters
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import os
import itertools
import six
import inspect
import abc
import logging

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from propeller.util import map_structure
from propeller.paddle.train import Saver
from propeller.types import InferenceSpec
from propeller.train.model import Model
from propeller.paddle.train.trainer import _build_net
from propeller.paddle.train.trainer import _build_model_fn
from propeller.types import RunMode
from propeller.types import ProgramPair

log = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Exporter(object):
    """base exporter"""

    @abc.abstractmethod
    def export(self, exe, program, eval_result, state):
        """export"""
        raise NotImplementedError()


class BestExporter(Exporter):
    """export saved model accordingto `cmp_fn`"""

    def __init__(self, export_dir, cmp_fn):
        """doc"""
        self._export_dir = export_dir
        self._best = None
        self.cmp_fn = cmp_fn

    def export(self, exe, program, eval_model_spec, eval_result, state):
        """doc"""
        log.debug('New evaluate result: %s \nold: %s' %
                  (repr(eval_result), repr(self._best)))
        if self._best is None and state['best_model'] is not None:
            self._best = state['best_model']
            log.debug('restoring best state %s' % repr(self._best))
        if self._best is None or self.cmp_fn(old=self._best, new=eval_result):
            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            eval_program = program.train_program
            # FIXME: all eval datasets has same name/types/shapes now!!! so every eval program are the smae

            saver = Saver(
                self._export_dir, exe, program=program, max_ckpt_to_keep=1)
            saver.save(state)
            eval_result = map_structure(float, eval_result)
            self._best = eval_result
            state['best_model'] = eval_result
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)


class BestInferenceModelExporter(Exporter):
    """export inference model accordingto `cmp_fn`"""

    def __init__(self,
                 export_dir,
                 cmp_fn,
                 model_class_or_model_fn=None,
                 hparams=None,
                 dataset=None):
        """doc"""
        self._export_dir = export_dir
        self._best = None
        self.cmp_fn = cmp_fn
        self.model_class_or_model_fn = model_class_or_model_fn
        self.hparams = hparams
        self.dataset = dataset

    def export(self, exe, program, eval_model_spec, eval_result, state):
        """doc"""
        if self.model_class_or_model_fn is not None and self.hparams is not None \
                and self.dataset is not None:
            log.info('Building program by user defined model function')
            if issubclass(self.model_class_or_model_fn, Model):
                _model_fn = _build_model_fn(self.model_class_or_model_fn)
            elif inspect.isfunction(self.model_class_or_model_fn):
                _model_fn = self.model_class_or_model_fn
            else:
                raise ValueError('unknown model %s' %
                                 self.model_class_or_model_fn)

            # build net
            infer_program = F.Program()
            startup_prog = F.Program()
            with F.program_guard(infer_program, startup_prog):
                #share var with Train net
                with F.unique_name.guard():
                    log.info('Building Infer Graph')
                    infer_fea = self.dataset.features()
                    # run_config is None
                    self.model_spec = _build_net(_model_fn, infer_fea,
                                                 RunMode.PREDICT, self.hparams,
                                                 None)
                    log.info('Done')
            infer_program = infer_program.clone(for_test=True)
            self.program = ProgramPair(
                train_program=infer_program, startup_program=startup_prog)

        else:
            self.program = program
            self.model_spec = eval_model_spec
        if self._best is None and state['best_inf_model'] is not None:
            self._best = state['best_inf_model']
            log.debug('restoring best state %s' % repr(self._best))
        log.debug('New evaluate result: %s \nold: %s' %
                  (repr(eval_result), repr(self._best)))

        if self._best is None or self.cmp_fn(old=self._best, new=eval_result):
            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            if self.model_spec.inference_spec is None:
                raise ValueError('model_fn didnt return InferenceSpec')

            inf_spec_dict = self.model_spec.inference_spec
            if not isinstance(inf_spec_dict, dict):
                inf_spec_dict = {'inference': inf_spec_dict}
            for inf_spec_name, inf_spec in six.iteritems(inf_spec_dict):
                if not isinstance(inf_spec, InferenceSpec):
                    raise ValueError('unknow inference spec type: %s' %
                                     inf_spec)

                save_dir = os.path.join(self._export_dir, inf_spec_name)
                log.debug('[Best Exporter]: save inference model: "%s" to %s' %
                          (inf_spec_name, save_dir))
                feed_var = [i.name for i in inf_spec.inputs]
                fetch_var = inf_spec.outputs

                infer_program = self.program.train_program
                startup_prog = F.Program()
                F.io.save_inference_model(
                    save_dir,
                    feed_var,
                    fetch_var,
                    exe,
                    main_program=infer_program)
            eval_result = map_structure(float, eval_result)
            state['best_inf_model'] = eval_result
            self._best = eval_result
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)

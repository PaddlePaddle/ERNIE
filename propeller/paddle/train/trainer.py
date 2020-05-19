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
"""common ML train and eval procedure"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import itertools
import six
import inspect
from collections import namedtuple
from contextlib import contextmanager
from six.moves import zip, map
import logging
from time import time

import paddle.fluid as F
import paddle.fluid.layers as L

from propeller.types import RunMode, StopException, SummaryRecord, StopException
from propeller.types import ModelSpec, InferenceSpec, ProgramPair, RunConfig
from propeller.paddle import summary, collection
from propeller.paddle.data.functional import Dataset
from propeller.paddle.train import distribution
from propeller.train.model import Model
from propeller.paddle.train.monitored_executor import Saver
from propeller.paddle.train import hooks, metrics

from propeller.paddle.train.monitored_executor import MonitoredExecutor

log = logging.getLogger(__name__)

__all__ = ['train_and_eval', 'Learner']


def _get_summary_writer(path):
    summary_writer = None
    try:
        from tensorboardX import SummaryWriter
        if distribution.status.is_master:
            summary_writer = SummaryWriter(os.path.join(path))
    except ImportError:
        log.warning('tensorboardX not installed, will not log to tensorboard')
    return summary_writer


def _get_one_place():
    return F.cuda_places()[0] if F.core.is_compiled_with_cuda(
    ) else F.cpu_places()[0]


def _log_eval_result(name, eval_result, swriter, state):
    log.debug(eval_result)
    printable = []
    for n, val in six.iteritems(eval_result):
        assert val.shape == (), 'metrics eval use float'
        printable.append('{}\t{}'.format(n, val))
        if swriter is not None:
            swriter.add_scalar(n, val, state.gstep)
            log.debug('write to tensorboard %s' % swriter.logdir)

    if len(printable):
        log.info('*** eval res: %10s ***' % name)
        for p in printable:
            log.info(p)
        log.info('******************************')


def _build_net(model_fn, features, mode, params, run_config):
    model_spec = model_fn(
        features=features, mode=mode, params=params, run_config=run_config)

    if mode == RunMode.TRAIN:
        if not isinstance(model_spec.loss, F.framework.Variable):
            raise ValueError('model_spec.metrics should be Variable, got %s' %
                             repr(model_spec.loss))
        if not (model_spec.loss.shape == () or model_spec.loss.shape == (1, )):
            raise ValueError('expect scarlar loss, got %s' %
                             repr(model_spec.loss.shape))
        model_spec.loss.persistable = True
    elif mode == RunMode.EVAL:
        if not isinstance(model_spec.metrics, dict):
            raise ValueError('model_spec.metrics should be dict, got %s' %
                             repr(model_spec.metrics))
    elif mode == RunMode.PREDICT:
        if not isinstance(model_spec.predictions, (list, tuple)):
            raise ValueError('model_spec.predictions shuold be list, got %s' %
                             repr(model_spec.predictions))
    else:
        raise ValueError('unkonw mode %s' % mode)
    return model_spec


class Learner(object):
    """A Learner can train / eval / predict on a Dataset"""

    def __init__(self,
                 model_class_or_model_fn,
                 run_config,
                 params=None,
                 warm_start_setting=None):
        """
        model_class_or_model_fn(callable|propeller.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of propeller.train.Model which implements:
                1. \_\_init\_\_       (hyper_param, mode, run_config)
                2. forward            (features) => (prediction)
                3. backword           (loss) => None
                4. loss               (predictoin) => (loss)
                5. metrics (optional) (prediction) => (dict of propeller.Metrics)
                
            2. a model_fn takes following args:
                1. features
                2. param
                3. mode
                4. run_config(optional)
               and returns a `propeller.ModelSpec`

        params: any python object, will pass to your `model_fn` or `propeller.train.Model`
        run_config (propeller.RunConfig): run_config.max_steps should not be None.
        warm_start_setting (propeller.WarmStartSetting): Optional. warm start variable will overwrite model variable.
        """
        if run_config.model_dir is None:
            raise ValueError('model_dir should specified in run_config')

        if issubclass(model_class_or_model_fn, Model):
            _model_fn = _build_model_fn(model_class_or_model_fn)
        elif inspect.isfunction(model_class_or_model_fn):
            _model_fn = model_class_or_model_fn
        else:
            raise ValueError('unknown model %s' % model_class_or_model_fn)

        self.model_fn = _model_fn
        self.params = params
        self.run_config = run_config
        self.warm_start_setting = warm_start_setting

    def _build_for_train(self, train_dataset):
        train_dataset.name = 'train'
        train_program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(train_program, startup_prog):
            with F.unique_name.guard():
                with collection.Collections() as collections:
                    log.info('Building Train Graph...')
                    fea = train_dataset.features()
                    model_spec = _build_net(self.model_fn, fea, RunMode.TRAIN,
                                            self.params, self.run_config)
                    log.info('Building Train Graph: Done')

                scalars = collections.get(collection.Key.SUMMARY_SCALAR)
                histograms = collections.get(collection.Key.SUMMARY_HISTOGRAM)
                skip_optimize_ops = collections.get(
                    collection.Key.SKIP_OPTIMIZE)
                skip_opt = set()
                if skip_optimize_ops is not None:
                    skip_opt |= set(skip_optimize_ops)
                if scalars is not None:
                    skip_opt |= {t for _, t in scalars}
                if histograms is not None:
                    skip_opt |= {t for _, t in histograms}
                skip_opt = list(skip_opt)
        log.info(
            'Train with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
            % (repr(self.run_config), repr(self.params), repr(model_spec)))

        summary_record = SummaryRecord(
            scalar=collections.get(collection.Key.SUMMARY_SCALAR),
            histogram=collections.get(collection.Key.SUMMARY_HISTOGRAM), )
        return ProgramPair(
            train_program=train_program,
            startup_program=startup_prog), model_spec, summary_record

    def _build_for_eval(self, ds):
        ds.name = 'eval'
        program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(program, startup_prog):
            #share var with Train net
            with F.unique_name.guard():
                log.info('Building Eval Graph')
                fea = ds.features()
                model_spec = _build_net(self.model_fn, fea, RunMode.EVAL,
                                        self.params, self.run_config)
                log.info('Done')
        program = program.clone(for_test=True)
        log.info(
            'Eval with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
            % (repr(self.run_config), repr(self.params), repr(model_spec)))
        return ProgramPair(
            train_program=program, startup_program=startup_prog), model_spec

    def _build_for_predict(self, ds):
        ds.name = 'predict'
        program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(program, startup_prog):
            #share var with Train net
            with F.unique_name.guard():
                log.info('Building Predict Graph')
                fea = ds.features()
                model_spec = _build_net(self.model_fn, fea, RunMode.PREDICT,
                                        self.params, self.run_config)
                log.info('Done')

        program = program.clone(for_test=True)

        log.info(
            'Predict with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
            % (repr(self.run_config), repr(self.params), repr(model_spec)))
        return ProgramPair(
            train_program=program, startup_program=startup_prog), model_spec

    def train(self, train_ds, train_hooks=[]):
        """train on a `Dataset`"""
        if not isinstance(train_ds, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(train_ds))

        train_program, model_spec, summary_record = self._build_for_train(
            train_ds)
        train_run_hooks = [
            hooks.StopAtStepHook(self.run_config.max_steps,
                                 self.run_config.run_steps),
            hooks.LoggingHook(
                model_spec.loss,
                summary_record=summary_record,
                summary_writer=_get_summary_writer(
                    os.path.join(self.run_config.model_dir, 'train_history')),
                per_step=self.run_config.log_steps,
                skip_step=self.run_config.skip_steps),
        ]
        if model_spec.train_hooks is not None:
            train_run_hooks.extend(model_spec.train_hooks)
        train_run_hooks.extend(train_hooks)

        train_executor = F.Executor(_get_one_place())

        mon_exe = MonitoredExecutor(
            train_executor,
            train_program,
            loss=model_spec.loss,
            run_config=self.run_config,
            run_hooks=train_run_hooks,
            warm_start_setting=self.warm_start_setting)

        distribution.init_distribuition_env(
            train_program)  #only initialize distribute training with 
        mon_exe.init_or_restore_variables()
        if distribution.status.is_master:
            mon_exe._hooks.append(
                hooks.CheckpointSaverHook(
                    mon_exe._saver,
                    per_step=mon_exe._save_steps,
                    skip_step=mon_exe._skip_steps))

        try:
            with mon_exe:
                for data in train_ds.start():
                    mon_exe.run(feed=data)
        except (StopException, F.core.EOFException) as e:
            pass

        return mon_exe.result

    def evaluate(self, eval_dataset, eval_hooks=[]):
        """eval on a `Dataset`"""
        if not isinstance(eval_dataset, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(eval_dataset))
        program, model_spec = self._build_for_eval(eval_dataset)
        single_card_place = _get_one_place()
        eval_executor = F.Executor(single_card_place)

        eval_run_hooks = [
            hooks.StopAtStepHook(self.run_config.eval_max_steps,
                                 self.run_config.eval_max_steps),
            hooks.EvalHook(model_spec.metrics, )
        ]

        if model_spec.eval_hooks is not None:
            eval_run_hooks.extend(model_spec.eval_hooks)
        eval_run_hooks.extend(eval_hooks)

        mon_exe = MonitoredExecutor(
            eval_executor,
            program,
            run_config=self.run_config,
            run_hooks=eval_run_hooks)
        mon_exe.init_or_restore_variables()

        try:
            with mon_exe:
                for data in eval_dataset.start(places=[single_card_place]):
                    mon_exe.run(feed=data)
        except (StopException, F.core.EOFException) as e:
            pass

        _, eval_result = mon_exe.result

        summary_writer = _get_summary_writer(
            os.path.join(self.run_config.model_dir, 'eval_history'))
        _log_eval_result('eval', eval_result, summary_writer, mon_exe.state)

        return mon_exe.result

    def predict(self,
                predict_dataset,
                ckpt=-1,
                ckpt_path=None,
                steps=-1,
                split_batch=True):
        """
        Perform predictoin
        will call `model_fn` and initiate user-specifed model in `propeller.RunMode.PREDICT` mode 

        Args:
            infer_dataset (propeller.data.Dataset): should not `shuffle` or `repeat`
            steps (int): steps to predict, if None is specifed, 
                will stop when `StopException` is raised in `infer_dataset`
            ckpt_path (None|str): Path of a specific checkpoint to predict. 
                If None, the latest checkpoint in model_dir is used. 
                If there are no checkpoints in model_dir, 
                prediction is run with newly initialized Variables instead of ones restored from checkpoint.
            ckpt (int): deprecated args
            split_batch (bool): if True, prediction of each example in a batch is returned.

        Yields:
            Evaluated values of predictions tensors.

        """
        if not isinstance(predict_dataset, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(predict_dataset))

        program, model_spec = self._build_for_predict(predict_dataset)
        single_card_place = _get_one_place()
        executor = F.Executor(single_card_place)
        pred_run_config = RunConfig(
            run_steps=steps if steps == -1 else None,
            model_dir=self.run_config.model_dir)
        mon_exe = MonitoredExecutor(
            executor,
            program,
            run_config=pred_run_config, )
        mon_exe.init_or_restore_variables(ckpt
                                          if ckpt_path is None else ckpt_path)
        try:
            with mon_exe:
                log.info('Runining predict from dir: %s' % repr(mon_exe.state))
                single_card_place = _get_one_place()
                for data in predict_dataset.start(places=[single_card_place]):
                    res = mon_exe.run(fetch_list=model_spec.predictions,
                                      feed=data)
                    if split_batch:
                        res = map(lambda i: i.tolist(), res)
                        res = zip(*res)  # transpose
                        for r in res:
                            yield r
                    else:
                        yield list(map(lambda i: i.tolist(), res))
        except (StopException, F.core.EOFException) as e:
            pass


def train_and_eval(_placeholder=None,
                   model_class_or_model_fn=None,
                   params=None,
                   run_config=None,
                   train_dataset=None,
                   eval_dataset=None,
                   warm_start_setting=None,
                   train_hooks=[],
                   eval_hooks=[],
                   exporters=[]):
    """
    Perform train and evaluate procesure. 
    will call `model_fn` and initiate user-specifed model in `propeller.RunMode.PREDICT` mode 

    Args:
        model_class_or_model_fn(callable|propeller.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of propeller.train.Model
            2. a model_fn takes following args: 1. features; 2. param; 3. mode; 4. run_config(optional)
               and returns a `propeller.ModelSpec`

        params: any python object, will pass to your `model_fn` or `propeller.train.Model`
        run_config (propeller.RunConfig): run_config.max_steps should not be None.
        train_dataset (propeller.paddle.data.Dataset): training will stop if global_step > run_config.max_steps.
        eval_dataset (propeller.paddle.data.Dataset|dict): Optional, if Dict of propeller.data.Dataset were specified, 
            will perform evluatation on every evaluation sets and report results.
        warm_start_setting (propeller.WarmStartSetting): Optional. warm start variable will overwrite model variable.
        train_hooks (list of propeller.paddle.train.RunHook): Optional.
        eval_hooks (list of propeller.paddle.train.RunHook): Optional.
        exporters (list of propeller.paddle.train.Exporter): Optional.
    """
    if _placeholder is not None:
        raise ValueError('specify keyword args to this function')
    if model_class_or_model_fn is None or params is None or run_config is None or train_dataset is None:
        raise ValueError(
            'some argument is None: model_class_or_model_fn:%s params:%s run_config:%s train_dataset:%s'
            % (model_class_or_model_fn, params, run_config, train_dataset))

    #init distribution env if envvir PROPELLER_DISCONFIG is set
    if train_dataset is None:
        raise ValueError('train dataset not specified')

    if eval_dataset is None:
        raise ValueError('eval dataset not specifed')

    if not isinstance(eval_dataset, (dict, Dataset)):
        raise ValueError(
            'Eval dataset should be propeller.Dataset of a list of that, got: %s'
            % eval_dataset)
    if isinstance(eval_dataset, Dataset):
        eval_dataset = {'eval': eval_dataset}
    ds_list = list(eval_dataset.values())
    for ds in ds_list:
        ds.name = 'eval'
    first = ds_list[0]
    for d in ds_list[1:]:
        if not first.__eq__(d):
            raise ValueError(
                'eval dataset has different output_shapes or types: %s' %
                repr(ds_list))

    est = Learner(
        model_class_or_model_fn,
        run_config,
        params,
        warm_start_setting=warm_start_setting)

    class _EvalHookOnTrainLoop(hooks.RunHook):
        def __init__(self):
            self.program, self.model_spec = est._build_for_eval(
                list(eval_dataset.values())[
                    0])  #eval_datasets must have same output shapes
            self.summary_writers = {
                ds_name: _get_summary_writer(
                    os.path.join(
                        os.path.join(run_config.model_dir, 'eval_history'),
                        ds_name))
                for ds_name in eval_dataset
            }

        def after_run(self, _, state):
            """doc"""
            if state.step > run_config.skip_steps and state.gstep % run_config.eval_steps == 0:
                eval_results = {}
                for name, ds in six.iteritems(eval_dataset):
                    ehooks = [
                        hooks.StopAtStepHook(est.run_config.eval_max_steps,
                                             est.run_config.eval_max_steps),
                        hooks.EvalHook(
                            self.model_spec.metrics,
                            summary_writer=self.summary_writers[name], )
                    ]
                    single_card_place = _get_one_place()
                    eval_executor = F.Executor(single_card_place)
                    mon_exe = MonitoredExecutor(
                        eval_executor,
                        self.program,
                        run_config=est.run_config,
                        run_hooks=ehooks + eval_hooks)
                    try:
                        with mon_exe:
                            for data in ds.start(places=[single_card_place]):
                                mon_exe.run(feed=data)
                    except (StopException, F.core.EOFException) as e:
                        pass
                    hook_results = mon_exe.result
                    eval_res = hook_results[
                        1]  # hook_results:  [StopAtStepHook, EvalHook, ...]
                    eval_results[name] = eval_res
                    _log_eval_result(name, eval_res,
                                     self.summary_writers[name], state)
                for exporter in exporters:
                    exporter.export(eval_executor, self.program,
                                    self.model_spec, eval_results, state)
            else:
                eval_results = {}
            return eval_results

    if distribution.status.is_master:
        train_hooks.append(_EvalHookOnTrainLoop())
    res = est.train(train_dataset, train_hooks=train_hooks)
    return res


def _build_model_fn(model_class):
    def _model_fn(features, mode, params, run_config):
        if mode != RunMode.PREDICT:
            fea, label = features[:-1], features[-1]
        else:
            fea = features

        model = model_class(params, mode, run_config=run_config)
        pred = model.forward(fea)
        if isinstance(pred, F.framework.Variable):
            prediction = [pred]
        else:
            prediction = pred
        if mode == RunMode.TRAIN:
            loss = model.loss(pred, label)
            model.backward(loss)
            return ModelSpec(loss=loss, predictions=prediction, mode=mode)
        elif mode == RunMode.EVAL:
            loss = model.loss(pred, label)
            me = model.metrics(pred, label)

            inf_spec = InferenceSpec(inputs=fea, outputs=prediction)
            if 'loss' not in me:
                me['loss'] = metrics.Mean(loss)
            return ModelSpec(
                loss=loss,
                predictions=prediction,
                metrics=me,
                mode=mode,
                inference_spec=inf_spec)
        elif mode == RunMode.PREDICT:
            inf_spec = InferenceSpec(inputs=fea, outputs=prediction)
            return ModelSpec(
                predictions=prediction, mode=mode, inference_spec=inf_spec)
        else:
            raise RuntimeError('unknown run mode %s' % mode)

    return _model_fn

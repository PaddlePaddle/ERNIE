#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""init model params"""

from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def cast_fp32_to_fp16(exe, main_program):
    """cast fp32 to fp16"""
    print("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            #load fp16
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.startswith("encoder_layer") \
                    and "layer_norm" not in param.name:
                print(param.name)
                param_t.set(np.float16(data).view(np.uint16), exe.place)
            
            #load fp32
            master_param_var = fluid.global_scope().find_var(param.name + 
                    ".master")
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """init model checkpoint"""
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path
    
    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program):
    """init pretraining params"""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))


def init_model(args, exe, startup_prog):
    """init model params"""
    init_func, init_path = None, None
    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_func = init_checkpoint
            init_path = args.init_checkpoint
        elif args.init_pretraining_params:
            init_func = init_pretraining_params
            init_path = args.init_pretraining_params

    elif args.do_val or args.do_test or args.do_pred:
        init_path = args.init_checkpoint or args.init_pretraining_params
        if not init_path:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_func = init_checkpoint
    if init_path:
        init_func(exe, init_path, main_program=startup_prog)

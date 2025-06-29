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

"""
Basic component
"""


import sys
from pathlib import Path

import gradio as gr

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from erniekit.webui import control
from erniekit.webui.common import config


def build(manager):
    """
    Basic component
    Args:
        manager (object): An object for unified management of components
    """

    default_basic_language = config.get_default_user_dict("basic", "language")
    default_basic_model_name = config.get_default_user_dict("basic", "model_name")
    default_basic_model_source = config.get_default_user_dict("basic", "model_source")
    default_basic_model_name_or_path = config.get_default_user_dict("basic", "model_name_or_path")
    default_basic_fine_tuning = config.get_default_user_dict("basic", "fine_tuning")
    default_basic_output_dir = config.get_default_user_dict("basic", "output_dir")
    default_basic_amp_master_grad = str(config.get_default_user_dict("basic", "amp_master_grad"))
    default_basic_compute_type = config.get_default_user_dict("basic", "compute_type")
    default_basic_tensor_parallel_degree = config.get_default_user_dict("basic", "tensor_parallel_degree")
    default_basic_pipeline_parallel_degree = config.get_default_user_dict("basic", "pipeline_parallel_degree")
    default_basic_sharding_parallel_degree = config.get_default_user_dict("basic", "sharding_parallel_degree")
    default_basic_pipeline_parallel_config = config.get_default_user_dict("basic", "pipeline_parallel_config")
    default_basic_pp_seg_method = config.get_default_user_dict("basic", "pp_seg_method")
    default_basic_use_sp_callback = str(config.get_default_user_dict("basic", "use_sp_callback"))
    default_basic_sharding = config.get_default_user_dict("basic", "sharding")
    default_basic_moe_group = config.get_default_user_dict("basic", "moe_group")
    default_basic_disable_ckpt_quant = str(config.get_default_user_dict("basic", "disable_ckpt_quant"))
    default_basic_lora_rank = config.get_default_user_dict("basic", "lora_rank")
    default_basic_lora_alpha = config.get_default_user_dict("basic", "lora_alpha")
    default_basic_lora_plus_scale = config.get_default_user_dict("basic", "lora_plus_scale")
    default_basic_rslora = str(config.get_default_user_dict("basic", "rslora"))

    with gr.Row(elem_classes="basic-info-row"):
        with gr.Row(scale=1):
            language = gr.Dropdown(
                choices=config.get_choices_kwargs("language"),
                value=default_basic_language,
            )

        with gr.Row(scale=2):
            model_name = gr.Dropdown(
                choices=config.get_choices_kwargs("model_name"),
                value=default_basic_model_name,
                interactive=False,
            )
        with gr.Row(scale=1):
            model_source = gr.Dropdown(
                choices=config.get_choices_kwargs("model_source_ernie"),
                value=default_basic_model_source,
                interactive=False,
            )

        with gr.Row(scale=3):
            model_name_or_path = gr.Textbox(
                value=default_basic_model_name_or_path,
            )

    with gr.Row(elem_classes="basic-info-row"):

        output_dir = gr.Textbox(
            visible=False,
            value=default_basic_output_dir,
        )

        output_dir_view = gr.Textbox(
            value=default_basic_output_dir,
        )

        gpu_num = gr.Number(
            value=config.get_gpu_count(),
            interactive=False,
        )

    with gr.Row(elem_classes="basic-info-row"):
        fine_tuning = gr.Dropdown(
            choices=config.get_choices_kwargs("fine_tuning"),
            value=default_basic_fine_tuning,
        )

        compute_type = gr.Dropdown(
            choices=config.get_choices_kwargs("compute_type_" + default_basic_fine_tuning),
            value=default_basic_compute_type,
        )

        amp_master_grad = gr.Dropdown(
            choices=config.get_choices_kwargs("boolean_choice"),
            value=default_basic_amp_master_grad,
        )
        disable_ckpt_quant = gr.Dropdown(
            choices=config.get_choices_kwargs("boolean_choice"),
            value=default_basic_disable_ckpt_quant,
        )

    with gr.Row(elem_classes="basic-info-row"):
        lora_rank = gr.Number(
            value=default_basic_lora_rank,
        )
        lora_alpha = gr.Number(
            value=default_basic_lora_alpha,
        )
        lora_plus_scale = gr.Number(
            value=default_basic_lora_plus_scale,
            precision=3,
        )
        rslora = gr.Dropdown(
            choices=config.get_choices_kwargs("boolean_choice"),
            value=default_basic_rslora,
        )
    with gr.Accordion(elem_classes="basic-info-row-accordion", open=False) as distributed_parameters_tab:
        with gr.Row():
            tensor_parallel_degree = gr.Number(
                value=default_basic_tensor_parallel_degree,
            )

            pipeline_parallel_degree = gr.Number(
                value=default_basic_pipeline_parallel_degree,
            )

            sharding_parallel_degree = gr.Number(
                value=default_basic_sharding_parallel_degree,
            )
        with gr.Row():

            pipeline_parallel_config = gr.Textbox(
                value=default_basic_pipeline_parallel_config,
            )

            pp_seg_method = gr.Textbox(
                value=default_basic_pp_seg_method,
            )

            sharding = gr.Textbox(
                value=default_basic_sharding,
            )

            use_sp_callback = gr.Dropdown(
                choices=config.get_choices_kwargs("boolean_choice"),
                value=default_basic_use_sp_callback,
            )

            moe_group = gr.Dropdown(
                choices=config.get_choices_kwargs("moe_group"),
                value=default_basic_moe_group,
            )

    manager.add_elem("basic", "language", language, default_basic_language)
    manager.add_elem("basic", "model_name", model_name, default_basic_model_name)
    manager.add_elem("basic", "model_source", model_source, default_basic_model_source)
    manager.add_elem("basic", "model_name_or_path", model_name_or_path, default_basic_model_name_or_path)
    manager.add_elem("basic", "compute_type", compute_type, default_basic_compute_type)
    manager.add_elem("basic", "fine_tuning", fine_tuning, default_basic_fine_tuning)
    manager.add_elem("basic", "output_dir", output_dir, default_basic_output_dir)
    manager.add_elem("basic", "output_dir_view", output_dir_view)

    manager.add_elem("basic", "amp_master_grad", amp_master_grad, default_basic_amp_master_grad)
    manager.add_elem(
        "basic",
        "tensor_parallel_degree",
        tensor_parallel_degree,
        default_basic_tensor_parallel_degree,
    )
    manager.add_elem(
        "basic",
        "pipeline_parallel_degree",
        pipeline_parallel_degree,
        default_basic_pipeline_parallel_degree,
    )
    manager.add_elem(
        "basic",
        "sharding_parallel_degree",
        sharding_parallel_degree,
        default_basic_sharding_parallel_degree,
    )

    manager.add_elem(
        "basic",
        "pipeline_parallel_config",
        pipeline_parallel_config,
        default_basic_pipeline_parallel_config,
    )
    manager.add_elem("basic", "pp_seg_method", pp_seg_method, default_basic_pp_seg_method)
    manager.add_elem("basic", "use_sp_callback", use_sp_callback, default_basic_use_sp_callback)
    manager.add_elem("basic", "sharding", sharding, default_basic_sharding)
    manager.add_elem("basic", "moe_group", moe_group, default_basic_moe_group)
    manager.add_elem(
        "basic",
        "disable_ckpt_quant",
        disable_ckpt_quant,
        default_basic_disable_ckpt_quant,
    )

    manager.add_elem("basic", "distributed_parameters_tab", distributed_parameters_tab)
    manager.add_elem("basic", "lora_rank", lora_rank, default_basic_lora_rank)
    manager.add_elem("basic", "lora_alpha", lora_alpha, default_basic_lora_alpha)
    manager.add_elem("basic", "lora_plus_scale", lora_plus_scale, default_basic_lora_plus_scale)
    manager.add_elem("basic", "rslora", rslora, default_basic_rslora)
    manager.add_elem("basic", "gpu_num", gpu_num)

    manager.add_module_dependency(
        source_module_id="basic",
        source_elem_id="best_config",
        update_module_id="basic",
        update_callback=control.model_update_callback,
        exclude_components=["language", "best_config"],
    )

    control.basic_reaction(manager)

    return language

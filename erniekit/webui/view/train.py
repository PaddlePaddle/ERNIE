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
Train component
"""

import gradio as gr

from erniekit.webui import control
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner


def build(manager):
    """
    Train component
    Args:
        manager (object): An object for unified management of components.
    """

    default_train_train_dataset_path = config.get_default_user_dict(
        "train", "train_dataset_path"
    )
    default_train_train_dataset_prob = config.get_default_user_dict(
        "train", "train_dataset_prob"
    )
    default_train_eval_dataset_path = config.get_default_user_dict(
        "train", "eval_dataset_path"
    )
    default_train_eval_dataset_prob = config.get_default_user_dict(
        "train", "eval_dataset_prob"
    )
    default_train_eval_dataset = config.get_default_user_dict("train", "eval_dataset")
    default_train_train_dataset = config.get_default_user_dict("train", "train_dataset")
    default_train_max_seq_len = config.get_default_user_dict("train", "max_seq_len")
    default_train_logging_dir = config.get_default_user_dict("train", "logging_dir")
    default_train_num_samples_each_epoch = config.get_default_user_dict(
        "train", "num_samples_each_epoch"
    )
    default_train_stage = config.get_default_user_dict("train", "stage")
    default_best_config = config.get_default_user_dict("basic", "best_config")
    default_train_num_train_epochs = config.get_default_user_dict(
        "train", "num_train_epochs"
    )
    default_train_gradient_accumulation_steps = config.get_default_user_dict(
        "train", "gradient_accumulation_steps"
    )
    default_train_max_steps = config.get_default_user_dict("train", "max_steps")
    default_train_batch_size = config.get_default_user_dict("train", "batch_size")
    default_train_recompute = str(config.get_default_user_dict("train", "recompute"))
    default_train_dataloader_num_workers = config.get_default_user_dict(
        "train", "dataloader_num_workers"
    )
    default_train_distributed_dataloader = str(
        config.get_default_user_dict("train", "distributed_dataloader")
    )
    default_train_learning_rate = config.get_default_user_dict("train", "learning_rate")
    default_train_lr_scheduler_type = config.get_default_user_dict(
        "train", "lr_scheduler_type"
    )
    default_train_min_lr = config.get_default_user_dict("train", "min_lr")
    default_train_layerwise_lr_decay_bound = config.get_default_user_dict(
        "train", "layerwise_lr_decay_bound"
    )
    default_train_weight_decay = config.get_default_user_dict("train", "weight_decay")
    default_train_adam_epsilon = float(
        config.get_default_user_dict("train", "adam_epsilon")
    )
    default_train_adam_beta1 = config.get_default_user_dict("train", "adam_beta1")
    default_train_adam_beta2 = config.get_default_user_dict("train", "adam_beta2")
    default_train_warmup_steps = config.get_default_user_dict("train", "warmup_steps")
    default_train_save_steps = config.get_default_user_dict("train", "save_steps")
    default_train_logging_steps = config.get_default_user_dict("train", "logging_steps")
    default_train_save_strategy = config.get_default_user_dict("train", "save_strategy")
    default_train_evaluation_strategy = config.get_default_user_dict(
        "train", "evaluation_strategy"
    )
    default_train_eval_steps = config.get_default_user_dict("train", "eval_steps")
    default_train_save_total_limit = config.get_default_user_dict(
        "train", "save_total_limit"
    )

    default_train_max_prompt_len = config.get_default_user_dict(
        "train", "max_prompt_len"
    )
    default_train_release_grads = str(
        config.get_default_user_dict("train", "release_grads")
    )
    default_train_offload_optim = str(
        config.get_default_user_dict("train", "offload_optim")
    )
    default_train_optim = config.get_default_user_dict("train", "optim")
    default_train_scale_loss = config.get_default_user_dict("train", "scale_loss")
    default_train_train_dataset_type = config.get_default_user_dict(
        "train", "train_dataset_type"
    )
    default_train_eval_dataset_type = config.get_default_user_dict(
        "train", "eval_dataset_type"
    )
    default_train_modality_ratio = config.get_default_user_dict(
        "train", "modality_ratio"
    )

    default_train_text_dataset_path = config.get_default_user_dict(
        "train", "text_dataset_path"
    )
    default_train_text_dataset_prob = config.get_default_user_dict(
        "train", "text_dataset_prob"
    )
    default_train_text_dataset_type = config.get_default_user_dict(
        "train", "text_dataset_type"
    )
    default_train_text_dataset = config.get_default_user_dict("train", "text_dataset")

    with gr.Tab() as train_tab:
        with gr.Row():
            best_config = gr.Dropdown(
                choices=config.get_choices_kwargs("best_config"),
                value=default_best_config,
            )

            max_seq_len = gr.Slider(
                minimum=4,
                maximum=131072,
                value=default_train_max_seq_len,
                step=1,
            )

            max_prompt_len = gr.Number(
                value=default_train_max_prompt_len,
                step=1,
            )

            num_samples_each_epoch = gr.Number(
                value=default_train_num_samples_each_epoch,
            )

            recompute = gr.Dropdown(
                choices=config.get_choices_kwargs("boolean_choice"),
                value=default_train_recompute,
            )

            logging_dir = gr.Textbox(value=default_train_logging_dir, visible=False)

        with gr.Row():
            stage = gr.Dropdown(
                choices=config.get_choices_kwargs("stages"),
                value=default_train_stage,
                interactive=False,
                visible=False,
            )

            num_train_epochs = gr.Number(
                value=default_train_num_train_epochs,
            )

            max_steps = gr.Number(
                value=default_train_max_steps,
            )

            batch_size = gr.Slider(
                minimum=1,
                maximum=1024,
                value=default_train_batch_size,
                step=1,
            )

            gradient_accumulation_steps = gr.Slider(
                minimum=1,
                value=default_train_gradient_accumulation_steps,
                maximum=1024,
                step=1,
            )

        with gr.Row() as train_train_dataset_row:
            train_dataset_path = gr.Textbox(
                visible=False,
                value=default_train_train_dataset_path,
            )

            train_dataset_prob = gr.Textbox(
                visible=False,
                value=default_train_train_dataset_prob,
            )

            train_dataset_type = gr.Textbox(
                visible=False,
                value=default_train_train_dataset_type,
            )

            with gr.Column():
                train_dataset_elem = control.create_dynamic_form_component(
                    manager=manager,
                    demo=manager.demo,
                    default_dataset=default_train_train_dataset,
                )

            with gr.Column():
                train_dataset_preview_btn = gr.Button()
                control.react_preview_dataset_button(
                    manager, train_dataset_preview_btn, "train", "train"
                )

        with gr.Row() as train_eval_dataset_row:
            eval_dataset_path = gr.Textbox(
                visible=False,
                value=default_train_eval_dataset_path,
            )

            eval_dataset_prob = gr.Textbox(
                visible=False,
                value=default_train_eval_dataset_prob,
            )

            eval_dataset_type = gr.Textbox(
                visible=False,
                value=default_train_eval_dataset_type,
            )
            with gr.Column():
                eval_dataset_elem = control.create_dynamic_form_component(
                    manager=manager,
                    demo=manager.demo,
                    default_dataset=default_train_eval_dataset,
                )
            with gr.Column():
                eval_dataset_preview_btn = gr.Button()
                control.react_preview_dataset_button(
                    manager, eval_dataset_preview_btn, "train", "eval"
                )

        with gr.Row() as train_text_dataset_row:
            text_dataset_path = gr.Textbox(
                visible=False,
                value=default_train_text_dataset_path,
            )

            text_dataset_prob = gr.Textbox(
                visible=False,
                value=default_train_text_dataset_prob,
            )

            text_dataset_type = gr.Textbox(
                visible=False,
                value=default_train_text_dataset_type,
            )
            with gr.Column():
                text_dataset_elem = control.create_dynamic_form_component(
                    manager=manager,
                    demo=manager.demo,
                    default_dataset=default_train_text_dataset,
                )
            with gr.Column():
                text_dataset_preview_btn = gr.Button()
                control.react_preview_dataset_button(
                    manager, text_dataset_preview_btn, "train", "text"
                )

        modality_ratio = gr.Slider(
            minimum=0,
            maximum=10,
            visible=False,
            step=1,
            value=default_train_modality_ratio,
        )

        with gr.Accordion(open=False) as dataloader_parameters_tab:
            with gr.Row():
                dataloader_num_workers = gr.Number(
                    value=default_train_dataloader_num_workers,
                )
                distributed_dataloader = gr.Dropdown(
                    choices=config.get_choices_kwargs("boolean_choice"),
                    value=default_train_distributed_dataloader,
                )

        with gr.Accordion(open=False) as optimizer_parameters_tab:
            with gr.Row():
                lr_scheduler_type = gr.Dropdown(
                    choices=[
                        "linear",
                        "cosine",
                        "polynomial",
                        "constant",
                        "constant_with_warmup",
                    ],
                    value=default_train_lr_scheduler_type,
                )
                learning_rate = gr.Number(
                    value=default_train_learning_rate,
                    step=1e-5,
                )
                min_lr = gr.Number(
                    value=default_train_min_lr,
                )
                layerwise_lr_decay_bound = gr.Slider(
                    minimum=0.001,
                    maximum=1,
                    value=default_train_layerwise_lr_decay_bound,
                    step=0.001,
                )
                warmup_steps = gr.Number(
                    value=default_train_warmup_steps,
                    step=1,
                )
            with gr.Row():
                optim = gr.Textbox(
                    label=default_train_optim,
                )
                offload_optim = gr.Dropdown(
                    choices=config.get_choices_kwargs("boolean_choice"),
                    value=default_train_offload_optim,
                )
                release_grads = gr.Dropdown(
                    choices=config.get_choices_kwargs("boolean_choice"),
                    value=default_train_release_grads,
                )
                scale_loss = gr.Number(
                    value=default_train_scale_loss,
                )
            with gr.Row():
                weight_decay = gr.Number(
                    value=default_train_weight_decay,
                    precision=3,
                )
                adam_epsilon = gr.Number(
                    value=float(default_train_adam_epsilon),
                    precision=8,
                    step=1e-8,
                )
                adam_beta1 = gr.Number(
                    value=default_train_adam_beta1,
                    precision=3,
                    step=0.1,
                )
                adam_beta2 = gr.Number(
                    value=default_train_adam_beta2,
                    precision=3,
                    step=0.001,
                )

        with gr.Accordion(open=True) as model_output_tab:
            with gr.Row():
                logging_steps = gr.Number(
                    value=default_train_logging_steps,
                    step=1,
                )
                eval_steps = gr.Number(
                    value=default_train_eval_steps,
                )

                evaluation_strategy = gr.Dropdown(
                    choices=config.get_choices_kwargs("strategy"),
                    value=default_train_evaluation_strategy,
                )

            with gr.Row():
                save_steps = gr.Number(
                    value=default_train_save_steps,
                    step=1,
                )

                save_strategy = gr.Dropdown(
                    choices=config.get_choices_kwargs("strategy"),
                    value=default_train_save_strategy,
                )

                save_total_limit = gr.Number(
                    value=default_train_save_total_limit,
                )

        with gr.Row():
            preview_command_btn = gr.Button()
            start_btn = gr.Button(variant="primary")
            stop_btn = gr.Button(variant="stop")

        with gr.Row():
            progress_display = gr.HTML(
                value="""
                <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 10px 0;">
                    <div style="width: 0%; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                        0%
                    </div>
                </div>
                """,
                label="进度显示",
                visible=False,
            )

        with gr.Row() as output_container:
            with gr.Column(scale=6):
                command_preview = gr.Code(
                    language="shell",
                    lines=15,
                    visible=False,
                    elem_classes="general-height-output textarea hide-copy-code",
                )

                output_text = gr.Textbox(
                    lines=15,
                    interactive=False,
                    elem_classes="general-height-output textarea",
                )

            with gr.Column(scale=4, visible=False) as output_plot_column:
                train_loss_plot = gr.LinePlot(
                    x="Step",
                    y="Loss",
                    title="Training Loss Curve",
                    x_title="Training Steps",
                    y_title="Loss Value",
                    width=600,
                    height=400,
                    color_map={"Loss": "blue"},
                )

        with gr.Row():
            clear_btn = gr.Button(variant="secondary")
            open_close_plot_btn = gr.Button(variant="secondary", visible=False)

    manager.add_elem("train", "preview_command_btn", preview_command_btn)
    manager.add_elem("train", "start_btn", start_btn)
    manager.add_elem("train", "stop_btn", stop_btn)
    manager.add_elem("train", "command_preview", command_preview)
    manager.add_elem("train", "open_close_plot_btn", open_close_plot_btn)

    manager.add_elem("train", "output_text", output_text)
    manager.add_elem("train", "output_container", output_container)
    manager.add_elem("train", "clean_btn", clear_btn)
    manager.add_elem("train", "logging_dir", logging_dir, default_train_logging_dir)
    manager.add_elem(
        "train",
        "train_dataset_type",
        train_dataset_type,
        default_train_train_dataset_type,
    )
    manager.add_elem(
        "train", "eval_dataset_type", eval_dataset_type, default_train_eval_dataset_type
    )
    manager.add_elem(
        "train",
        "train_dataset_path",
        train_dataset_path,
        default_train_train_dataset_path,
    )
    manager.add_elem(
        "train",
        "train_dataset_prob",
        train_dataset_prob,
        default_train_train_dataset_prob,
    )

    manager.add_elem(
        "train", "eval_dataset_path", eval_dataset_path, default_train_eval_dataset_path
    )
    manager.add_elem(
        "train", "eval_dataset_prob", eval_dataset_prob, default_train_eval_dataset_prob
    )
    manager.add_elem("train", "max_seq_len", max_seq_len, default_train_max_seq_len)
    manager.add_elem(
        "train",
        "num_samples_each_epoch",
        num_samples_each_epoch,
        default_train_num_samples_each_epoch,
    )
    manager.add_elem("basic", "best_config", best_config, default_best_config)
    manager.add_elem("train", "stage", stage, default_train_stage)
    manager.add_elem(
        "train", "num_train_epochs", num_train_epochs, default_train_num_train_epochs
    )
    manager.add_elem(
        "train",
        "gradient_accumulation_steps",
        gradient_accumulation_steps,
        default_train_gradient_accumulation_steps,
    )
    manager.add_elem("train", "max_steps", max_steps, default_train_max_steps)
    manager.add_elem("train", "batch_size", batch_size, default_train_batch_size)
    manager.add_elem("train", "recompute", recompute, default_train_recompute)
    manager.add_elem(
        "train",
        "dataloader_num_workers",
        dataloader_num_workers,
        default_train_dataloader_num_workers,
    )
    manager.add_elem(
        "train",
        "distributed_dataloader",
        distributed_dataloader,
        default_train_distributed_dataloader,
    )
    manager.add_elem(
        "train", "learning_rate", learning_rate, default_train_learning_rate
    )
    manager.add_elem(
        "train", "lr_scheduler_type", lr_scheduler_type, default_train_lr_scheduler_type
    )
    manager.add_elem("train", "min_lr", min_lr, default_train_min_lr)
    manager.add_elem(
        "train",
        "layerwise_lr_decay_bound",
        layerwise_lr_decay_bound,
        default_train_layerwise_lr_decay_bound,
    )
    manager.add_elem("train", "weight_decay", weight_decay, default_train_weight_decay)
    manager.add_elem(
        "train", "adam_epsilon", adam_epsilon, float(default_train_adam_epsilon)
    )
    manager.add_elem("train", "adam_beta1", adam_beta1, default_train_adam_beta1)
    manager.add_elem("train", "adam_beta2", adam_beta2, default_train_adam_beta2)
    manager.add_elem("train", "warmup_steps", warmup_steps, default_train_warmup_steps)
    manager.add_elem("train", "save_steps", save_steps, default_train_save_steps)
    manager.add_elem(
        "train", "logging_steps", logging_steps, default_train_logging_steps
    )
    manager.add_elem(
        "train",
        "save_strategy",
        save_strategy,
        default_train_save_strategy,
    )
    manager.add_elem(
        "train",
        "evaluation_strategy",
        evaluation_strategy,
        default_train_evaluation_strategy,
    )
    manager.add_elem("train", "eval_steps", eval_steps, default_train_eval_steps)
    manager.add_elem(
        "train", "save_total_limit", save_total_limit, default_train_save_total_limit
    )
    manager.add_elem(
        "train", "max_prompt_len", max_prompt_len, default_train_max_prompt_len
    )
    manager.add_elem(
        "train", "release_grads", release_grads, default_train_release_grads
    )
    manager.add_elem(
        "train", "offload_optim", offload_optim, default_train_offload_optim
    )
    manager.add_elem("train", "optim", optim, default_train_optim)
    manager.add_elem("train", "scale_loss", scale_loss, default_train_scale_loss)
    manager.add_elem("train", "train_tab", train_tab)
    manager.add_elem("train", "clear_btn", clear_btn)
    manager.add_elem("train", "dataloader_parameters_tab", dataloader_parameters_tab)
    manager.add_elem("train", "optimizer_parameters_tab", optimizer_parameters_tab)
    manager.add_elem("train", "model_output_tab", model_output_tab)
    manager.add_elem("train", "train_dataset_preview_btn", train_dataset_preview_btn)
    manager.add_elem("train", "train_dataset_group", train_dataset_elem["output"])
    manager.add_elem(
        "train", "train_dataset_save_btn", train_dataset_elem["save_dataset_btn"]
    )
    manager.add_elem("train", "train_dataset_btn", train_dataset_elem["dataset_btn"])
    manager.add_elem("train", "eval_dataset_group", eval_dataset_elem["output"])
    manager.add_elem(
        "train", "eval_dataset_save_btn", eval_dataset_elem["save_dataset_btn"]
    )
    manager.add_elem("train", "eval_dataset_btn", eval_dataset_elem["dataset_btn"])
    manager.add_elem("train", "eval_dataset_preview_btn", eval_dataset_preview_btn)
    manager.add_elem("train", "progress_display", progress_display)
    manager.add_elem("train", "train_loss_plot", train_loss_plot)
    manager.add_elem("train", "output_plot_column", output_plot_column)
    manager.add_elem(
        "train", "modality_ratio", modality_ratio, default_train_modality_ratio
    )

    manager.add_elem(
        "train", "text_dataset_path", text_dataset_path, default_train_text_dataset_path
    )
    manager.add_elem(
        "train", "text_dataset_prob", text_dataset_prob, default_train_text_dataset_prob
    )
    manager.add_elem(
        "train", "text_dataset_type", text_dataset_type, default_train_text_dataset_type
    )
    manager.add_elem("train", "text_dataset_preview_btn", text_dataset_preview_btn)
    manager.add_elem(
        "train", "text_dataset_save_btn", text_dataset_elem["save_dataset_btn"]
    )
    manager.add_elem("train", "text_dataset_btn", text_dataset_elem["dataset_btn"])
    manager.add_elem("train", "text_dataset_group", text_dataset_elem["output"])

    manager.add_elem("train", "text_dataset_row", train_text_dataset_row)
    manager.add_elem("train", "eval_dataset_row", train_eval_dataset_row)
    manager.add_elem("train", "train_dataset_row", train_train_dataset_row)

    manager.add_module_dependency(
        source_module_id="basic",
        source_elem_id="best_config",
        update_module_id="train",
        update_callback=control.model_update_callback,
        exclude_components=[
            "start_btn",
            "preview_command_btn",
            "stop_btn",
            "command_preview",
            "output_text",
            "output_container",
            "clean_btn",
            "train_dataset_preview_btn",
        ],
    )
    control.train_vl_reaction_for_dataset_row(
        manager,
        train_dataset_elem["row_components"],
        eval_dataset_elem["row_components"],
        text_dataset_elem["row_components"],
    )
    control.train_reaction(manager, CommandRunner(), "train")
    control.train_update_by_basic_model_name_group(
        manager, train_dataset_elem, eval_dataset_elem, text_dataset_elem
    )

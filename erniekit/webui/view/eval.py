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
Eval component
"""


import gradio as gr

from erniekit.webui import control
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner


def build(manager):
    """
    Eval component
    Args:
        manager (object): An object for unified management of components
    """

    default_eval_eval_dataset_path = config.get_default_user_dict("eval", "eval_dataset_path")
    default_eval_eval_dataset_prob = config.get_default_user_dict("eval", "eval_dataset_prob")
    default_eval_eval_customize_dataset_type = config.get_default_user_dict("eval", "eval_customize_dataset_type")
    default_eval_eval_existed_dataset_path = config.get_default_user_dict("eval", "eval_existed_dataset_path")

    default_eval_max_seq_len = config.get_default_user_dict("eval", "max_seq_len")
    default_eval_batch_size = config.get_default_user_dict("eval", "batch_size")
    default_eval_eval_dataset = config.get_default_user_dict("eval", "eval_dataset")
    default_eval_eval_dataset_type = config.get_default_user_dict("eval", "eval_dataset_type")
    default_eval_eval_existed_dataset_prob = config.get_default_user_dict("eval", "eval_existed_dataset_prob")
    default_eval_eval_existed_dataset_type = config.get_default_user_dict("eval", "eval_existed_dataset_type")
    default_eval_logging_dir = config.get_default_user_dict("eval", "logging_dir")

    with gr.Tab() as eval_tab:

        with gr.Row():
            eval_dataset_path = gr.Textbox(
                value=default_eval_eval_dataset_path,
                lines=2,
                visible=False,
            )
            eval_dataset_prob = gr.Textbox(
                value=default_eval_eval_dataset_prob,
                lines=1,
                visible=False,
            )
            eval_dataset_type = gr.Textbox(value=default_eval_eval_dataset_type, visible=False)

            logging_dir = gr.Textbox(value=default_eval_logging_dir, visible=False)

            with gr.Accordion(open=False) as eval_builtin_dataset_tab:
                with gr.Column(scale=1):
                    eval_dataset = gr.Dropdown(
                        choices=config.get_choices_kwargs("existed_dataset_list"),
                        multiselect=True,
                        value=default_eval_eval_dataset,
                    )

                    eval_existed_dataset_path = gr.Textbox(
                        elem_classes="dataset-text-height-output",
                        value=default_eval_eval_existed_dataset_path,
                    )

                    with gr.Row():
                        with gr.Row():
                            eval_existed_dataset_prob = gr.Textbox(
                                value=default_eval_eval_existed_dataset_prob,
                            )

                        with gr.Row():
                            eval_existed_dataset_type = gr.Textbox(value=default_eval_eval_existed_dataset_type)

                    eval_existed_preview_btn = gr.Button()

                    control.react_preview_dataset_button(manager, eval_existed_preview_btn, "eval", "eval_existed")

            with gr.Accordion(open=False) as eval_customize_dataset_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Row(scale=4):
                                eval_customize_dataset_type = gr.Textbox(
                                    value=default_eval_eval_customize_dataset_type,
                                )

                            with gr.Row(scale=2):
                                eval_customize_select_dataset_type = gr.Dropdown(
                                    value=None,
                                    choices=config.get_choices_kwargs("dataset_type"),
                                )

                        eval_customize_dataset_path = gr.Textbox(
                            lines=2,
                            value=default_eval_eval_dataset_path,
                        )

                        eval_customize_dataset_prob = gr.Textbox(
                            value=default_eval_eval_dataset_prob,
                        )

                        eval_customize_preview_btn = gr.Button()

                        control.react_preview_dataset_button(
                            manager, eval_customize_preview_btn, "eval", "eval_customize"
                        )

        with gr.Row():

            max_seq_len = gr.Slider(
                minimum=1024,
                maximum=131072,
                value=default_eval_max_seq_len,
                step=1,
            )

            batch_size = gr.Slider(
                minimum=1,
                maximum=1024,
                value=default_eval_batch_size,
                step=1,
            )

        with gr.Row():
            preview_command_btn = gr.Button()
            start_btn = gr.Button(variant="primary")
            stop_btn = gr.Button(variant="stop")

        with gr.Column() as output_container:

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

        clear_btn = gr.Button(variant="secondary")

    manager.add_elem("eval", "preview_command_btn", preview_command_btn)
    manager.add_elem("eval", "start_btn", start_btn)
    manager.add_elem("eval", "stop_btn", stop_btn)
    manager.add_elem("eval", "command_preview", command_preview)
    manager.add_elem("eval", "output_text", output_text)
    manager.add_elem("eval", "output_container", output_container)
    manager.add_elem("eval", "clear_btn", clear_btn)
    manager.add_elem("eval", "eval_customize_preview_btn", eval_customize_preview_btn)
    manager.add_elem("eval", "eval_existed_preview_btn", eval_existed_preview_btn)
    manager.add_elem("eval", "eval_builtin_dataset_tab", eval_builtin_dataset_tab)
    manager.add_elem("eval", "eval_customize_dataset_tab", eval_customize_dataset_tab)

    manager.add_elem("eval", "eval_customize_select_dataset_type", eval_customize_select_dataset_type)
    manager.add_elem("eval", "logging_dir", logging_dir, default_eval_logging_dir)
    manager.add_elem(
        "eval", "eval_existed_dataset_type", eval_existed_dataset_type, default_eval_eval_existed_dataset_type
    )
    manager.add_elem("eval", "eval_dataset_type", eval_dataset_type, default_eval_eval_existed_dataset_type)
    manager.add_elem("eval", "eval_dataset_path", eval_dataset_path, default_eval_eval_dataset_path)
    manager.add_elem("eval", "eval_dataset_prob", eval_dataset_prob, default_eval_eval_dataset_prob)

    manager.add_elem(
        "eval", "eval_existed_dataset_path", eval_existed_dataset_path, default_eval_eval_existed_dataset_path
    )
    manager.add_elem(
        "eval", "eval_existed_dataset_prob", eval_existed_dataset_prob, default_eval_eval_existed_dataset_prob
    )
    manager.add_elem(
        "eval", "eval_customize_dataset_path", eval_customize_dataset_path, default_eval_eval_dataset_path
    )
    manager.add_elem(
        "eval", "eval_customize_dataset_prob", eval_customize_dataset_prob, default_eval_eval_dataset_prob
    )
    manager.add_elem(
        "eval", "eval_customize_dataset_type", eval_customize_dataset_type, default_eval_eval_customize_dataset_type
    )

    manager.add_elem("eval", "eval_dataset", eval_dataset, default_eval_eval_dataset)
    manager.add_elem("eval", "max_seq_len", max_seq_len, default_eval_max_seq_len)
    manager.add_elem("eval", "batch_size", batch_size, default_eval_batch_size)
    manager.add_elem("eval", "eval_tab", eval_tab)

    manager.add_specific_elem_by_id("eval", "eval_dataset", eval_dataset, default_eval_eval_dataset)

    control.eval_reaction(manager, CommandRunner(), "eval")

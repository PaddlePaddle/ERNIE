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
Export component
"""

import gradio as gr

from erniekit.webui import control
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner


def build(manager):
    """
    Export component
    Args:
        manager (object): An object for unified management of components
    """

    default_max_shard_size = config.get_default_user_dict("export", "max_shard_size")

    with gr.Tab() as export_tab:
        with gr.Row():
            max_shard_size = gr.Slider(
                minimum=1,
                maximum=100,
                value=default_max_shard_size,
                step=1,
            )

        with gr.Row():
            preview_command_btn = gr.Button()
            start_merge_btn = gr.Button(
                variant="primary",
            )
            start_split_btn = gr.Button(variant="primary")

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

    manager.add_elem("export", "preview_command_btn", preview_command_btn)
    manager.add_elem("export", "start_merge_btn", start_merge_btn)
    manager.add_elem("export", "start_split_btn", start_split_btn)
    manager.add_elem("export", "stop_btn", stop_btn)
    manager.add_elem("export", "command_preview", command_preview)
    manager.add_elem("export", "output_text", output_text)
    manager.add_elem("export", "output_container", output_container)
    manager.add_elem("export", "clear_btn", clear_btn)
    manager.add_elem("export", "start_merge_btn", start_merge_btn)
    manager.add_elem("export", "start_split_btn", start_split_btn)
    manager.add_elem("export", "max_shard_size", max_shard_size, default_max_shard_size)
    manager.add_elem("export", "export_tab", export_tab)

    control.export_reaction(manager, CommandRunner(), "export")

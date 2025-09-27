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
Chat component
"""

import gradio as gr

from erniekit.webui import control
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner


def build(manager):
    """
    Chat component
    Args:
        manager (object): An object for unified management of components.
    """

    default_port = config.get_default_user_dict("chat", "port")
    default_max_model_len = config.get_default_user_dict("chat", "max_model_len")
    default_max_new_tokens = config.get_default_user_dict("chat", "max_new_tokens")
    default_top_p = config.get_default_user_dict("chat", "top_p")
    default_temperature = config.get_default_user_dict("chat", "temperature")
    default_role_setting = config.get_default_user_dict("chat", "role_setting")
    default_system_prompt = config.get_default_user_dict("chat", "system_prompt")
    default_thought_checkbox = config.get_default_user_dict("chat", "thought_checkbox")

    with gr.Tab() as chat_tab:
        chat_info = gr.HTML()

        with gr.Row():
            progress_display = gr.HTML(
                value="""
                <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 10px 0;">
                    <div style="width: 0%; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                        0%
                    </div>
                </div>
                """,
                visible=False,
            )

        with gr.Row():
            with gr.Column(scale=7):
                output_text = gr.Textbox(
                    interactive=False,
                    elem_classes="chat-height-output",
                )

            with gr.Column(scale=3):
                max_model_len = gr.Slider(
                    minimum=1024,
                    maximum=131072,
                    value=default_max_model_len,
                    step=1,
                )

                port = gr.Number(value=default_port)

                save_port = gr.Number(value=default_port, visible=False)

                status_button = gr.Button(
                    variant="secondary",
                )

        with gr.Row():
            with gr.Column():
                load_model_btn = gr.Button(variant="primary")

            # 功能键
            with gr.Column():
                unload_model_btn = gr.Button(variant="stop")

        with gr.Row():
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    height=550,
                    show_label=True,
                    show_copy_button=True,
                    type="messages",
                )

            with gr.Column(scale=2):
                with gr.Column():
                    max_new_tokens = gr.Slider(
                        minimum=1024,
                        maximum=131072,
                        value=default_max_new_tokens,
                        step=1,
                    )

                    top_p = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=default_top_p,
                        step=0.01,
                    )

                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.5,
                        value=default_temperature,
                        step=0.01,
                    )

                with gr.Column():
                    thought_checkbox = gr.Checkbox(
                        visible=False,
                        elem_classes="large-checkbox",
                        value=default_thought_checkbox,
                    )

                submit_btn = gr.Button()

                stop_btn = gr.Button()

                clear_btn = gr.Button()

                download_log_btn = gr.DownloadButton(visible=False)
                generate_log_btn = gr.DownloadButton()

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column() as chat_tab_col:
                    chat_input = gr.Textbox(
                        lines=3, show_copy_button=True, container=True
                    )
                with gr.Column() as file_tab_col:

                    file_input = gr.File(
                        file_count="multiple",
                        file_types=config.get_file_type(),
                        visible=False,
                        elem_classes="custom-file-input",
                    )

                    img_url_input = gr.Textbox(
                        lines=2,
                        visible=False,
                    )

                    video_url_input = gr.Textbox(
                        lines=2,
                        visible=False,
                    )

            with gr.Row():
                role_setting = gr.Textbox(
                    lines=2,
                )
                system_prompt = gr.Textbox(lines=2)

        with gr.Column():
            thinking_display = gr.Markdown(value="")

            response_display = gr.Markdown(value="")

    manager.add_elem("chat", "status_button", status_button)
    manager.add_elem("chat", "thinking_display", thinking_display)
    manager.add_elem("chat", "response_display", response_display)
    manager.add_elem("chat", "submit_btn", submit_btn)
    manager.add_elem("chat", "output_text", output_text)
    manager.add_elem("chat", "chat_input", chat_input)
    manager.add_elem("chat", "chatbot", chatbot)
    manager.add_elem("chat", "clear_btn", clear_btn)
    manager.add_elem("chat", "load_model_btn", load_model_btn)
    manager.add_elem("chat", "unload_model_btn", unload_model_btn)
    manager.add_elem("chat", "stop_btn", stop_btn)
    manager.add_elem("chat", "chat_info", chat_info)
    manager.add_elem("chat", "progress_display", progress_display)

    manager.add_elem("chat", "port", port, default_port)
    manager.add_elem("chat", "save_port", save_port, default_port)
    manager.add_elem("chat", "max_model_len", max_model_len, default_max_model_len)
    manager.add_elem("chat", "max_new_tokens", max_new_tokens, default_max_new_tokens)
    manager.add_elem("chat", "top_p", top_p, default_top_p)
    manager.add_elem("chat", "temperature", temperature, default_temperature)
    manager.add_elem("chat", "role_setting", role_setting, default_role_setting)
    manager.add_elem("chat", "system_prompt", system_prompt, default_system_prompt)
    manager.add_elem("chat", "file_input", file_input)
    manager.add_elem("chat", "thought_checkbox", thought_checkbox, False)
    manager.add_elem("chat", "download_log_btn", download_log_btn)
    manager.add_elem("chat", "generate_log_btn", generate_log_btn)
    manager.add_elem("chat", "img_url_input", img_url_input)
    manager.add_elem("chat", "video_url_input", video_url_input)
    manager.add_elem("chat", "chat_tab_col", chat_tab_col)
    manager.add_elem("chat", "file_tab_col", file_tab_col)

    manager.add_elem("chat", "chat_tab", chat_tab)

    control.chat_reaction(manager, CommandRunner())

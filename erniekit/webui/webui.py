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
The startup interface of ErnieKit WebUI
"""

import os
import resource
import sys
from pathlib import Path

import gradio as gr
import logging

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
webui_dir = Path(__file__).parent
sys.path.insert(0, str(webui_dir))

from alert import alert  # noqa: E402
from common import config  # noqa: E402
from manager import manager  # noqa: E402
from view import basic, chat, eval, export, train  # noqa: E402
from view.style import CSS, html_log  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)


def is_env_enabled(env_name: str) -> bool:
    """Check if environment variables are enabled"""
    return os.getenv(env_name, "").lower() in ["true", "1", "yes"]


def fix_proxy(ipv6_enabled: bool = False) -> None:
    """Fix Gradio UI proxy settings to prevent local connections from being disrupted by proxies"""
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    if ipv6_enabled:
        os.environ.pop("http_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTPS_PROXY", None)


def create_ui():
    with gr.Blocks(title="ErnieKit WebUI", theme=gr.themes.Ocean()) as demo:
        gr.HTML(f"<style>{CSS}</style>")
        if config.get_paddle_png():
            gr.HTML(html_log.format(config.get_paddle_png()))

        manager.demo = demo
        language = basic.build(manager)

        with gr.Tabs(elem_classes="large-tabs"):
            train.build(manager)
            chat.build(manager)
            eval.build(manager)
            export.build(manager)

        if language:
            manager.setup_language_switching(language, demo, alert)

        manager.setup_component_tracking(demo)
    return demo


def run_webui():
    # Read environment variable configuration
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")

    # Fix proxy settings to avoid network interference
    fix_proxy(ipv6_enabled=gradio_ipv6)

    print("Starting ErnieKit WebUI")

    demo = create_ui()
    demo.queue().launch(server_name=server_name, share=gradio_share, inbrowser=True)


if __name__ == "__main__":
    run_webui()

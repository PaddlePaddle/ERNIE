# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the code for the chatbot demo using Gradio."""

import argparse
import base64
import json
import logging
import os
from argparse import ArgumentParser
from collections import namedtuple
from functools import partial

import gradio as gr
from bot_requests import BotClient

os.environ["NO_PROXY"] = "localhost,127.0.0.1"  # Disable proxy

logging.root.setLevel(logging.INFO)

MULTI_MODEL_PREFIX = "ERNIE-4.5-VL"


def get_args() -> argparse.Namespace:
    """
    Parses and returns command line arguments for configuring the chatbot demo.
    Sets up argument parser with default values for server configuration and model endpoints.
    The arguments include:
    - Server port and name for the Gradio interface
    - Character limits and retry settings for conversation handling
    - Model name to endpoint mappings for the chatbot

    Returns:
        argparse.Namespace: Parsed command line arguments containing all the above settings
    """
    parser = ArgumentParser(description="ERNIE models web chat demo.")

    parser.add_argument("--server-port", type=int, default=8232, help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--max_char", type=int, default=8000, help="Maximum character limit for messages.")
    parser.add_argument("--max_retry_num", type=int, default=3, help="Maximum retry number for request.")
    parser.add_argument(
        "--model_map",
        type=str,
        required=True,
        default="""{
            "ERNIE-4.5-300B-A47B": "http://localhost:port/v1",
            "ERNIE-4.5-21B-A3B": "http://localhost:port/v1",
            "ERNIE-4.5-0.3B": "http://localhost:port/v1",
            "ERNIE-4.5-VL-424B-A47B": "http://localhost:port/v1",
            "ERNIE-4.5-VL-28B-A3B": "http://localhost:port/v1"
        }""",
        help="""JSON string defining model name to endpoint mappings.
            Required Format:
            {"model_name": "http://localhost:port/v1", ...}

            Note:
            - All endpoints must be valid HTTP URLs
            - At least one model must be specified
            - Prefix determines model capabilities:
            * ERNIE-4.5[-*]: Text-only model
            * ERNIE-4.5-VL[-*]: Multimodal models (image+text)
            """,
    )

    args = parser.parse_args()
    try:
        args.model_map = json.loads(args.model_map)

        # Validation: Check at least one model exists
        if len(args.model_map) < 1:
            raise ValueError("model_map must contain at least one model configuration")
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for --model-map") from e

    return args


class GradioEvents:
    """
    Central handler for all Gradio interface events in the chatbot demo. Provides static methods
    for processing user interactions including:
    - Response regeneration
    - Conversation state management
    - Image handling and URL conversion
    - Component visibility control

    Coordinates with BotClient to interface with backend models while maintaining
    conversation history and handling multimodal inputs.
    """

    @staticmethod
    def get_image_url(image_path: str) -> str:
        """
        Converts an image file at the given path to a base64 encoded data URL
        that can be used directly in HTML or Gradio interfaces.
        Reads the image file, encodes it in base64 format, and constructs
        a data URL with the appropriate image MIME type.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Image URL.
        """
        base64_image = ""
        extension = image_path.split(".")[-1]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/{extension};base64,{base64_image}"
        return url

    @staticmethod
    def chat_stream(
        query: str,
        task_history: list,
        image_history: dict,
        model_name: str,
        file_url: str,
        system_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        bot_client: BotClient,
    ) -> str:
        """
        Handles streaming chat interactions by processing user queries and
        generating real-time responses from the bot client. Constructs conversation
        history including system messages, text inputs and image attachments, then
        streams back model responses.

        Args:
            query (str): User input.
            task_history (list): Task history.
            image_history (dict): Image history.
            model_name (str): Model name.
            file_url (str): File URL.
            system_msg (str): System message.
            max_tokens (int): Maximum tokens.
            temperature (float): Temperature.
            top_p (float): Top p.
            bot_client (BotClient): Bot client.

        Yields:
            str: Model response.
        """
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        for idx, (query_h, response_h) in enumerate(task_history):
            if idx in image_history:
                content = []
                content.append(
                    {"type": "image_url", "image_url": {"url": GradioEvents.get_image_url(image_history[idx])}}
                )
                content.append({"type": "text", "text": query_h})
                conversation.append({"role": "user", "content": content})
            else:
                conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})

        content = []
        if file_url and (len(image_history) == 0 or file_url != list(image_history.values())[-1]):
            image_history[len(task_history)] = file_url
            content.append({"type": "image_url", "image_url": {"url": GradioEvents.get_image_url(file_url)}})
            content.append({"type": "text", "text": query})
            conversation.append({"role": "user", "content": content})
        else:
            conversation.append({"role": "user", "content": query})

        try:
            req_data = {"messages": conversation}
            for chunk in bot_client.process_stream(model_name, req_data, max_tokens, temperature, top_p):
                if "error" in chunk:
                    raise Exception(chunk["error"])

                message = chunk.get("choices", [{}])[0].get("delta", {})
                content = message.get("content", "")

                if content:
                    yield content

        except Exception as e:
            raise gr.Error("Exception: " + repr(e))

    @staticmethod
    def predict_stream(
        query: str,
        chatbot: list,
        task_history: list,
        image_history: dict,
        model: str,
        file_url: str,
        system_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        bot_client: BotClient,
    ) -> list:
        """
        Processes user queries in a streaming manner by coordinating with the chat stream handler,
        progressively updates the chatbot state with responses,
        and maintains conversation history. Handles both text and multimodal inputs while preserving
        the interactive chat experience with real-time updates.

        Args:
            query (str): The user's query.
            chatbot (list): The current chatbot state.
            task_history (list): The task history.
            image_history (dict): The image history.
            model (str): The model name.
            file_url (str): The file URL.
            system_msg (str): The system message.
            max_tokens (int): The maximum token length of the generated response.
            temperature (float): The temperature parameter used by the model.
            top_p (float): The top_p parameter used by the model.
            bot_client (BotClient): The bot client.

        Returns:
            list: A list containing the updated chatbot state after processing the user's query.
        """

        logging.info(f"User: {query}")
        chatbot.append({"role": "user", "content": query})

        # First yield the chatbot with user message
        yield chatbot

        new_texts = GradioEvents.chat_stream(
            query, task_history, image_history, model, file_url, system_msg, max_tokens, temperature, top_p, bot_client
        )

        response = ""
        for new_text in new_texts:
            response += new_text

            # Remove previous message if exists
            if chatbot[-1].get("role") == "assistant":
                chatbot.pop(-1)

            if response:
                chatbot.append({"role": "assistant", "content": response})
                yield chatbot

        logging.info(f"History: {task_history}")
        task_history.append((query, response))
        logging.info(f"ERNIE models: {response}")

    @staticmethod
    def regenerate(
        chatbot: list,
        task_history: list,
        image_history: dict,
        model: str,
        file_url: str,
        system_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        bot_client: BotClient,
    ) -> list:
        """
        Reconstructs the conversation context by removing the last interaction and
        reprocesses the user's previous query to generate a fresh response. Maintains
        consistency in conversation flow while allowing response regeneration.

        Args:
            chatbot (list): The current chatbot state.
            task_history (list): The task history.
            image_history (dict): The image history.
            model (str): The model name.
            file_url (str): The file URL.
            system_msg (str): The system message.
            max_tokens (int): The maximum token length of the generated response.
            temperature (float): The temperature parameter used by the model.
            top_p (float): The top_p parameter used by the model.
            bot_client (BotClient): The bot client.

        Yields:
            list: A list containing the updated chatbot state after processing the user's query.
        """
        if not task_history:
            yield chatbot
            return
        # Pop the last user query and bot response from task_history
        item = task_history.pop(-1)
        if (len(task_history)) in image_history:
            del image_history[len(task_history)]
        while len(chatbot) != 0 and chatbot[-1].get("role") == "assistant":
            chatbot.pop(-1)
        chatbot.pop(-1)

        yield from GradioEvents.predict_stream(
            item[0],
            chatbot,
            task_history,
            image_history,
            model,
            file_url,
            system_msg,
            max_tokens,
            temperature,
            top_p,
            bot_client,
        )

    @staticmethod
    def reset_user_input() -> gr.update:
        """
        Reset user input field value to empty string.

        Returns:
            gr.update: Update object representing the new value of the user input field.
        """
        return gr.update(value="")

    @staticmethod
    def reset_state() -> tuple:
        """
        Reset all states including chatbot, task_history, image_history, and file_btn.

        Returns:
            tuple: A tuple containing the following values:
                - chatbot (list): An empty list that represents the cleared chatbot state.
                - task_history (list): An empty list that represents the cleared task history.
                - image_history (dict): An empty dictionary that represents the cleared image history.
                - file_btn (gr.update): An update object that sets the value of the file button to None.
        """
        GradioEvents.gc()

        reset_result = namedtuple("reset_result", ["chatbot", "task_history", "image_history", "file_btn"])
        return reset_result(
            [],  # clear chatbot
            [],  # clear task_history
            {},  # clear image_history
            gr.update(value=None),  # clear file_btn
        )

    @staticmethod
    def gc():
        """Run garbage collection to free up memory resources."""
        import gc

        gc.collect()

    @staticmethod
    def toggle_components_visibility(model_name: str) -> gr.update:
        """
        Toggle visibility of components depending on the selected model name.

        Args:
            model_name (str): Name of the selected model.

        Returns:
            gr.update: An update object representing the visibility of the file button.
        """
        return gr.update(visible=model_name.upper().startswith(MULTI_MODEL_PREFIX))  # file_btn


def launch_demo(args: argparse.Namespace, bot_client: BotClient):
    """
    Launch demo program

    Args:
        args (argparse.Namespace): argparse Namespace object containing parsed command line arguments
        bot_client (BotClient): Bot client instance
    """
    css = """
    /* Hide original Chinese text */
    #file-upload .wrap {
        font-size: 0 !important;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Insert English prompt text below the SVG icon */
    #file-upload .wrap::after {
        content: "Drag and drop files here or click to upload";
        font-size: 15px;
        color: #555;
        white-space: nowrap;
    }
    """
    with gr.Blocks(css=css) as demo:
        logo_url = GradioEvents.get_image_url("assets/logo.png")
        gr.Markdown(
            f"""\
                <p align="center"><img src="{logo_url}" \
                style="height: 60px"/><p>"""
        )
        gr.Markdown(
            """\
<center><font size=3>This demo is based on ERNIE models. \
(本演示基于文心大模型实现。)</center>"""
        )

        chatbot = gr.Chatbot(label="ERNIE", elem_classes="control-height", type="messages")
        model_names = list(args.model_map.keys())
        with gr.Row():
            model_name = gr.Dropdown(
                label="Select Model", choices=model_names, value=model_names[0], allow_custom_value=True
            )
            file_btn = gr.File(
                label="Image upload (Active only for multimodal models. Accepted formats: PNG, JPEG, JPG)",
                height="80px",
                visible=True,
                file_types=[".png", ".jpeg", "jpg"],
                elem_id="file-upload",
            )
        query = gr.Textbox(label="Input", elem_id="text_input")

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History(清除历史)")
            submit_btn = gr.Button("🚀 Submit(发送)", elem_id="submit-button")
            regen_btn = gr.Button("🤔️ Regenerate(重试)")

        with gr.Accordion("⚙️ Advanced Config", open=False):  # open=False means collapsed by default
            system_message = gr.Textbox(value="", label="System message", visible=True)
            additional_inputs = [
                system_message,
                gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Top-p (nucleus sampling)"),
            ]

        task_history = gr.State([])
        image_history = gr.State({})

        model_name.change(GradioEvents.toggle_components_visibility, inputs=model_name, outputs=file_btn)
        model_name.change(
            GradioEvents.reset_state, outputs=[chatbot, task_history, image_history, file_btn], show_progress=True
        )
        predict_with_clients = partial(GradioEvents.predict_stream, bot_client=bot_client)
        regenerate_with_clients = partial(GradioEvents.regenerate, bot_client=bot_client)
        query.submit(
            predict_with_clients,
            inputs=[query, chatbot, task_history, image_history, model_name, file_btn] + additional_inputs,
            outputs=[chatbot],
            show_progress=True,
        )
        query.submit(GradioEvents.reset_user_input, [], [query])
        submit_btn.click(
            predict_with_clients,
            inputs=[query, chatbot, task_history, image_history, model_name, file_btn] + additional_inputs,
            outputs=[chatbot],
            show_progress=True,
        )
        submit_btn.click(GradioEvents.reset_user_input, [], [query])
        empty_btn.click(
            GradioEvents.reset_state, outputs=[chatbot, task_history, image_history, file_btn], show_progress=True
        )
        regen_btn.click(
            regenerate_with_clients,
            inputs=[chatbot, task_history, image_history, model_name, file_btn] + additional_inputs,
            outputs=[chatbot],
            show_progress=True,
        )

        demo.load(GradioEvents.toggle_components_visibility, inputs=gr.State(model_names[0]), outputs=file_btn)

    demo.queue().launch(server_port=args.server_port, server_name=args.server_name)


def main():
    """Main function that runs when this script is executed."""
    args = get_args()
    bot_client = BotClient(args)
    launch_demo(args, bot_client)


if __name__ == "__main__":
    main()

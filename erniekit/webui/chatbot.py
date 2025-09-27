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
Optimized Chatbot architecture supporting text, multimodal, and thought models
with enhanced debugging capabilities
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import gradio as gr
import openai

from erniekit.webui.common import config
from erniekit.webui.alert import alert


@dataclass
class ChatRequest:
    """Chat request configuration"""

    message: str
    model_name: str
    history: List[Dict[str, str]]
    file_input: Optional[List[str]] = None
    url_input: str = None
    role_setting: Optional[str] = None
    system_prompt: Optional[str] = None
    max_length: int = 1000
    top_p: float = 0.8
    temperature: float = 0.7
    port: int = 8188
    enable_thinking: bool = False


@dataclass
class DebugInfo:
    """Debug information for each chat turn"""

    timestamp: str
    session_id: str
    model_type: str
    request_data: Dict[str, Any]
    requested_messages: List[Dict[str, str]]
    response_content: str
    thought_content: str = ""
    generation_time: float = 0.0
    token_count: int = 0
    error_info: Optional[str] = None


class DebugLogger:
    """Debug logging and output system"""

    def __init__(self, enabled: bool = False, log_level: str = "INFO"):
        self.enabled = enabled
        self.session_logs: Dict[str, List[DebugInfo]] = {}
        self.current_session_id: Optional[str] = None

        # Setup logging
        # logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)

    def enable_debug(self, session_id: Optional[str] = None):
        """
        Enable debug logging for this session

        Args:
            self: Instance reference
            session_id (str): session id
        """
        self.enabled = True
        if session_id:
            self.current_session_id = session_id
        else:
            self.current_session_id = f"session_{int(time.time())}"

        if self.current_session_id not in self.session_logs:
            self.session_logs[self.current_session_id] = []

        self.logger.info(f"Debug mode enabled for session: {self.current_session_id}")

    def disable_debug(self):
        """
        Disable debug logging for this session

        Args:
        self: Instance reference
        """
        self.enabled = False
        self.logger.info("Debug mode disabled")

    def log_debug_info(self, debug_info: DebugInfo):
        """
        Log debug information for the current session

        Args:
        self: Instance reference
        debug_info (DebugInfo): Debug information object to be logged
        """

        if not self.enabled or not self.current_session_id:
            return

        self.session_logs[self.current_session_id].append(debug_info)

        # Print debug info
        # self._print_debug_info(debug_info)

    def _print_debug_info(self, debug_info: DebugInfo):
        """
        Print formatted debug information

        Args:
        self: Instance reference
        debug_info (DebugInfo): Debug information object to be printed in formatted form
        """

        print("\n" + "=" * 80)
        print(f"🐛 DEBUG INFO - {debug_info.timestamp}")
        print("=" * 80)
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"🤖 Model Type: {debug_info.model_type}")
        print(f"⏱️  Generation Time: {debug_info.generation_time:.2f}s")
        print(f"🔢 Token Count: {debug_info.token_count}")

        print("\n📥 REQUEST DATA:")
        print("-" * 40)
        print(f"Message: {debug_info.request_data.get('message', 'N/A')}")
        print(f"File Input: {debug_info.request_data.get('file_input', 'N/A')}")
        print(f"Role Setting: {debug_info.request_data.get('role_setting', 'N/A')}")
        print(f"System Prompt: {debug_info.request_data.get('system_prompt', 'N/A')}")
        print(f"Temperature: {debug_info.request_data.get('temperature', 'N/A')}")
        print(f"Top P: {debug_info.request_data.get('top_p', 'N/A')}")
        print(f"Max Length: {debug_info.request_data.get('max_length', 'N/A')}")
        print(f"Port: {debug_info.request_data.get('port', 'N/A')}")
        print(
            f"Enable Thinking: {debug_info.request_data.get('enable_thinking', 'N/A')}"
        )

        print("\n📨 PROCESSED MESSAGES:")
        print("-" * 40)
        for i, msg in enumerate(debug_info.requested_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content_preview = f"[Multimodal content with {len(content)} parts]"
            else:
                content_preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"  {i + 1}. [{role.upper()}]: {content_preview}")

        if debug_info.thought_content:
            print("\n🤔 THOUGHT PROCESS:")
            print("-" * 40)
            thought_preview = debug_info.thought_content[:300] + (
                "..." if len(debug_info.thought_content) > 300 else ""
            )
            print(thought_preview)

        print("\n💬 RESPONSE CONTENT:")
        print("-" * 40)
        response_preview = debug_info.response_content[:300] + (
            "..." if len(debug_info.response_content) > 300 else ""
        )
        print(response_preview)

        if debug_info.error_info:
            print("\n❌ ERROR INFO:")
            print("-" * 40)
            print(debug_info.error_info)

        print("=" * 80 + "\n")

    def get_session_logs(self, session_id: Optional[str] = None) -> List[DebugInfo]:
        """
        Get logs for a specific session or the current session if no session ID is provided

        Args:
        self: Instance reference
        session_id (Optional[str]): Session ID to retrieve logs for. If None, uses current session ID
        Returns:
        List[DebugInfo]: List of debug information objects for the specified session
        """

        target_session = session_id or self.current_session_id
        return self.session_logs.get(target_session, [])

    def export_session_logs(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """
        Export logs for a specific session or the current session if no session ID is provided

        Args:
        self: Instance reference
        session_id (Optional[str]): Session ID to export logs for. If None, uses current session ID
        format (str): Export format ('json' or 'txt')
        Returns:
        str: Formatted string representation of the logs
        """
        logs = self.get_session_logs(session_id)

        if format.lower() == "json":
            return json.dumps(
                [asdict(log) for log in logs], ensure_ascii=False, indent=2
            )
        elif format.lower() == "txt":
            output = []
            for log in logs:
                output.append(f"Timestamp: {log.timestamp}")
                output.append(f"Model Type: {log.model_type}")
                output.append(f"Request: {log.request_data.get('message', '')}")
                output.append(f"Response: {log.response_content}")
                output.append("-" * 50)
            return "\n".join(output)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_session_logs(self, session_id: Optional[str] = None):
        """
        Clear logs for a specific session or the current session if no session ID is provided

        Args:
        self: Instance reference
        session_id (Optional[str]): Session ID to clear logs for. If None, uses current session ID
        """
        target_session = session_id or self.current_session_id
        if target_session in self.session_logs:
            del self.session_logs[target_session]
            self.logger.info(f"Cleared logs for session: {target_session}")


class ModelType:
    """Model type constants"""

    TEXT = "text"
    MULTIMODAL = "multimodal"
    THOUGHT = "thought"
    MULTIMODAL_THOUGHT = "multimodal_thought"


class MessageProcessor:
    """Handle message history processing and formatting"""

    @staticmethod
    async def build_message_history(
        message: str,
        history: List[Union[Dict, List, Tuple]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_multimodal: bool = False,
        file_input: Optional[List[str]] = None,
        url_input: str = None,
        chatbot_instance=None,
    ) -> List[Dict[str, Any]]:
        """
        Build standardized message history from various input formats

        Args:
            message: Current user message
            history: Conversation history in various formats
            role_setting: Role configuration
            system_prompt: System prompt
            is_multimodal: Whether this is for multimodal model
            file_input: List of file paths
            chatbot_instance: ChatBotGenerator instance for file classification

        Returns:
            Standardized message list
        """
        messages = []

        system_content = MessageProcessor._build_system_content(
            role_setting, system_prompt
        )
        if system_content:
            messages.append({"role": "system", "content": system_content})

        if history:
            messages.extend(MessageProcessor._parse_history(history, is_multimodal))

        if is_multimodal:
            user_content = MessageProcessor._parse_multimodal_content(
                message, file_input, chatbot_instance, url_input
            )
        else:
            user_content = message

        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _build_system_content(
        role_setting: Optional[str], system_prompt: Optional[str]
    ) -> str:
        """
        Build system content based on role setting and system prompt

        Args:
            role_setting: Role configuration
            system_prompt: System prompt

        Returns:
            String representing system content
        """
        content_parts = []
        if role_setting:
            content_parts.append(
                alert.get("role_setting", "append", config.get_language()).format(
                    role_setting
                )
            )
        if system_prompt:
            content_parts.append(system_prompt)
        return "".join(content_parts)

    @staticmethod
    def _parse_history(
        history: List[Union[Dict, List, Tuple]], is_multimodal: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Parse conversation history into a standardized list of dictionaries

        Args:
        history (List[Union[Dict, List, Tuple]]): Conversation history data in various formats
        is_multimodal (bool, optional): Flag indicating if history contains multimodal content. Defaults to False
        Returns:
        List[Dict[str, Any]]: Standardized list of dictionaries representing conversation history

        """
        messages = []

        for entry in history:
            if isinstance(entry, dict) and "role" in entry:
                role = entry["role"]
                content = entry.get("content", "")
                if role in ["user", "assistant"]:
                    if is_multimodal and role == "user" and isinstance(content, str):
                        content = MessageProcessor._parse_multimodal_content(content)
                    messages.append({"role": role, "content": content})
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                user_msg, bot_msg = entry
                user_content = user_msg
                if is_multimodal:
                    user_content = MessageProcessor._parse_multimodal_content(user_msg)
                messages.append({"role": "user", "content": user_content})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
            else:
                print(f"Warning: Unresolvable history format: {entry}")

        return messages

    @staticmethod
    def _parse_multimodal_content(
        message: str,
        file_input: Optional[List[str]] = None,
        chatbot_instance=None,
        url_input: str = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Parse message for multimodal content (local images, videos) and text

        Args:
            message: Text message
            file_input: List of local file paths
            chatbot_instance: ChatBotGenerator instance for file classification

        Returns:
            Either text string or list of content dictionaries
        """
        content_list = []
        LOCAL_FILE_PREFIX = "file://"

        if file_input and chatbot_instance:
            classified_files = chatbot_instance._classifie_file_by_ext(file_input)

            for img_path in classified_files.get("image_url", []):
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{LOCAL_FILE_PREFIX}{img_path}",
                        },
                    }
                )

            for vid_path in classified_files.get("video_url", []):
                content_list.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"{LOCAL_FILE_PREFIX}{vid_path}"},
                    }
                )
        if url_input and chatbot_instance:
            img_url_input = url_input["image"]
            for img_path in img_url_input.split(";"):
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_path},
                    }
                )
            video_url_input = url_input["video"]
            for video_path in video_url_input.split(";"):
                content_list.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": video_path},
                    }
                )

        if message.strip():
            content_list.append({"type": "text", "text": message.strip()})

        if (
            content_list
            and len(content_list) > 1
            or (content_list and content_list[0]["type"] != "text")
        ):
            return content_list

        return message


class ResponseFormatter:
    """Format different types of responses"""

    @staticmethod
    def format_thought_response(thought_content: str, response_content: str) -> str:
        """
        Format thought response with HTML details element

        Args:
            thought_content: Thought content as a string
            response_content: Response content as a string

        Returns:
            Formatted HTML string
        """
        thought_process = alert.get("thought_process", "append", config.get_language())
        return (
            f"<details open><summary>{thought_process}</summary>\n"
            f"<div class='thought-container' style='font-size: 13px;opacity: 0.85;"
            f"padding-left:20px;border-left:3px solid #ddd;"
            f"margin-bottom: 1em;'>\n{thought_content}</div>\n"
            f"</details>\n"
            f"<div class='response-container' style='line-height: 1.5;'>{response_content}</div>"
        )


class BaseResponseGenerator(ABC):
    """Base class for response generators"""

    def __init__(self, client_factory, debug_logger: DebugLogger):
        self.client_factory = client_factory
        self.debug_logger = debug_logger
        self.stop_generation = False

    def stop(self):
        """
        Stop the current generation process

        Args:
            self: Instance reference

        """
        self.stop_generation = True

    def reset(self):
        """
        Reset the generator state

        Args:
            self: Instance reference
        """
        self.stop_generation = False

    @abstractmethod
    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Abstract method to generate a response to a chat request

        Args:
            self: Instance reference
            request (ChatRequest): The chat request object containing input parameters
            chatbot_instance: Optional reference to the chatbot instance, defaults to None

        """

        pass

    async def _create_chat_completion(
        self,
        client,
        messages: List[Dict],
        request: ChatRequest,
        enable_thinking: bool = False,
    ):
        """
        Create a chat completion using the given client and messages

        Args:
            client: OpenAI API client
            messages (List[Dict]): List of chat messages
            request (ChatRequest): Chat request object
            enable_thinking (bool, optional): Whether to enable thinking mode. Defaults to False
        """

        if enable_thinking:
            request_messages = []
            for message in messages:
                request_message = message.copy()
                request_message["chat_template_kwargs"] = {
                    "enable_thinking": enable_thinking
                }
                request_messages.append(request_message)
        else:
            request_messages = messages

        completion_params = {
            "model": request.model_name,
            "messages": request_messages,
            "stream": True,
        }

        return client.chat.completions.create(**completion_params)

    def _create_history_with_response(
        self, request: ChatRequest
    ) -> Tuple[List[Dict], Dict]:
        """
        Create conversation history including the response to the given request

        Args:
        self: Instance reference
        request (ChatRequest): The chat request object to process

        """
        new_history = list(request.history) if request.history else []

        user_message = {"role": "user", "content": request.message}
        new_history.append(user_message)

        assistant_response = {"role": "assistant", "content": ""}
        new_history.append(assistant_response)

        return new_history, assistant_response

    def _create_debug_info(
        self, request: ChatRequest, messages: List[Dict], model_type: str
    ) -> DebugInfo:
        """
        Create debug information object for the chat session

        Args:
            self: Instance reference
            request (ChatRequest): Chat request object containing input parameters
            messages (List[Dict]): List of message dictionaries in the chat
            model_type (str): Type of model being used for the chat

        Returns:
            DebugInfo: Debug information object containing session details
        """
        return DebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=self.debug_logger.current_session_id or "unknown",
            model_type=model_type,
            request_data=asdict(request),
            requested_messages=messages,
            response_content="",
            thought_content="",
            generation_time=0.0,
            token_count=0,
        )


class TextResponseGenerator(BaseResponseGenerator):
    """Generator for text-only models"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate text response for a chat request

        Args:
            self: Instance reference
            request (ChatRequest): Chat request object containing input parameters
            chatbot_instance: Optional chatbot instance reference, defaults to None

        """
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=False,
                file_input=request.file_input,
                url_input=request.url_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(request, messages, ModelType.TEXT)

            response = await self._create_chat_completion(client, messages, request)
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            token_count = 0
            async for chunk in self._stream_response(response, assistant_response):
                if self.stop_generation:
                    break
                token_count += 1
                yield new_history, gr.update(value="")
                await asyncio.sleep(0.01)

            # Update debug info
            debug_info.response_content = assistant_response["content"]
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _stream_response(self, response, assistant_response):
        """
        Stream the response generated by the API call

        Args:
            self: Instance reference
            response: Response object returned by the API call
            assistant_response (Dict): Assistant response dictionary
        """
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                assistant_response["content"] += chunk.choices[0].delta.content
                yield chunk

    async def _handle_error(self, error: Exception, message: str):
        """
        Handle errors during response generation

        Args:
            self: Instance reference
            error (Exception): Error that occurred during response generation
            message (str): User's original message
        """
        print(f"Text response error: {error}")
        error_msg = {
            "role": "assistant",
            "content": alert.get("chatbot_api", "text", config.get_language()).format(
                error
            ),
        }
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class MultimodalResponseGenerator(BaseResponseGenerator):
    """Generator for multimodal models"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate multimodal response for a chat request

        Args:
            self: Instance reference
            request (ChatRequest): Chat request object containing input parameters
            chatbot_instance: Optional chatbot instance reference, defaults to None
        """
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=True,
                file_input=request.file_input,
                url_input=request.url_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(
                request, messages, ModelType.MULTIMODAL
            )

            response = await self._create_chat_completion(client, messages, request)
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            token_count = 0
            async for chunk in self._stream_response(response, assistant_response):
                if self.stop_generation:
                    break
                token_count += 1
                yield new_history, gr.update(value="")
                await asyncio.sleep(0.01)

            # Update debug info
            debug_info.response_content = assistant_response["content"]
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _stream_response(self, response, assistant_response):
        """
        Stream the assistant's response incrementally

        Args:
            self: Instance reference
            response: The response object from the model/API
            assistant_response: Object to accumulate or handle the streaming response
        """
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                assistant_response["content"] += chunk.choices[0].delta.content
                yield chunk

    async def _handle_error(self, error: Exception, message: str):
        """
        Handle exceptions and errors that occur during processing

        Args:
            self: Instance reference
            error (Exception): The exception object that was raised
            message (str): Additional error message or context for the error
        """
        print(f"Multimodal response error: {error}")
        error_msg = {
            "role": "assistant",
            "content": alert.get(
                "chatbot_api", "multimodal", config.get_language()
            ).format(error),
        }
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class ThoughtResponseGenerator(BaseResponseGenerator):
    """Generator for models with thought process"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Handle exceptions and errors that occur during processing.
        This includes logging the error details, formatting user-friendly messages,
        and potentially performing cleanup operations.

        Args:
            self: Instance reference
            error (Exception): The exception object that was raised
            message (str): Additional error message or context for the error
        """
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=False,
                file_input=request.file_input,
                url_input=request.url_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(request, messages, ModelType.THOUGHT)

            response = await self._create_chat_completion(
                client, messages, request, enable_thinking=True
            )
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            current_thought = ""
            current_response = ""
            token_count = 0

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta:
                    thought_part = (
                        getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
                    )
                    answer_part = getattr(chunk.choices[0].delta, "content", "") or ""

                    current_thought += thought_part
                    current_response += answer_part
                    token_count += 1

                    formatted_response = ResponseFormatter.format_thought_response(
                        current_thought, current_response
                    )

                    assistant_response["content"] = formatted_response
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            debug_info.thought_content = current_thought
            debug_info.response_content = current_response
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _handle_error(self, error: Exception, message: str):
        """
        Handle exceptions and errors that occur during processing.
        This includes logging the error details, formatting user-friendly messages,
        and potentially performing cleanup operations.

        Args:
            self: Instance reference
            error (Exception): The exception object that was raised
            message (str): Additional error message or context for the error
        """
        print(f"Thought response error: {error}")
        error_msg = {
            "role": "assistant",
            "content": alert.get(
                "chatbot_api", "thought", config.get_language()
            ).format(error),
        }
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class MultimodalThoughtResponseGenerator(BaseResponseGenerator):
    """Generator for multimodal models with thought process"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate a response using multimodal models with integrated thought process

        Args:
            self: Instance reference
            request (ChatRequest): Chat request object containing input parameters,
                which may include text, images, or other multimodal content
            chatbot_instance: Optional chatbot instance reference, defaults to None

        Returns:
            AsyncGenerator yielding tuples containing:
                - List of dictionaries representing chat messages with multimodal content
                - gr.update object for interface updates
        """
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=True,
                file_input=request.file_input,
                url_input=request.url_input,
                chatbot_instance=chatbot_instance,
            )

            debug_info = self._create_debug_info(
                request, messages, ModelType.MULTIMODAL_THOUGHT
            )

            response = await self._create_chat_completion(
                client, messages, request, enable_thinking=True
            )
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            current_thought = ""
            current_response = ""
            token_count = 0

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta:
                    thought_part = (
                        getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
                    )
                    answer_part = getattr(chunk.choices[0].delta, "content", "") or ""

                    current_thought += thought_part
                    current_response += answer_part
                    token_count += 1

                    formatted_response = ResponseFormatter.format_thought_response(
                        current_thought, current_response
                    )

                    assistant_response["content"] = formatted_response
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            debug_info.thought_content = current_thought
            debug_info.response_content = current_response
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _handle_error(self, error: Exception, message: str):
        """
        Handle errors encountered during multimodal response generation

        Args:
            self: Instance reference
            error (Exception): The exception object that was raised
            message (str): Contextual message describing where/why the error occurred

        Handles:
            - Logging error details for debugging purposes
            - Generating user-friendly error messages
            - Ensuring proper cleanup of resources if needed
            - Propagating or suppressing errors based on configuration
        """
        print(f"Multimodal thought response error: {error}")
        error_msg = {
            "role": "assistant",
            "content": alert.get(
                "chatbot_api", "multimodal_thought", config.get_language()
            ).format(error),
        }

        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class ChatBotGenerator:
    """
    Main ChatBot class supporting multiple model types with debug capabilities:
    - Text models
    - Multimodal models
    - Thought models
    - Multimodal thought models
    """

    def __init__(
        self, debug_enabled: bool = False, debug_session_id: Optional[str] = None
    ):
        self.default_ip = "0.0.0.0"
        self.debug_logger = DebugLogger()

        if debug_enabled:
            self.enable_debug(debug_session_id)

        self.generators = self._initialize_generators()

    def _initialize_generators(self) -> Dict[str, BaseResponseGenerator]:
        """
        Initialize response generators for different model types

        Creates and configures appropriate appropriate response generator instances
        for various model types (e.g., text-only, multimodal, etc.)

        Returns:
            Dict[str, BaseResponseGenerator]: A dictionary mapping model type identifiers
            to their corresponding BaseResponseGenerator instances
        """
        client_factory = self._create_openai_client

        return {
            ModelType.TEXT: TextResponseGenerator(client_factory, self.debug_logger),
            ModelType.MULTIMODAL: MultimodalResponseGenerator(
                client_factory, self.debug_logger
            ),
            ModelType.THOUGHT: ThoughtResponseGenerator(
                client_factory, self.debug_logger
            ),
            ModelType.MULTIMODAL_THOUGHT: MultimodalThoughtResponseGenerator(
                client_factory, self.debug_logger
            ),
        }

    def _create_openai_client(self, port: int) -> openai.Client:
        """
        Create and configure an OpenAI client connection

        Args:
            self: Instance reference
            port (int): Port number to use for the OpenAI client connection

        Returns:
            openai.Client: Configured OpenAI client instance ready for API interactions
        """
        base_url = f"http://{self.default_ip}:{port}/v1"
        return openai.Client(base_url=base_url, api_key="null")

    def _classifie_file_by_ext(self, file_input: List) -> Dict:
        """
        Classify input files by their extensions

        Args:
            self: Instance reference
            file_input (List): List containing file-related items (e.g., file paths, file objects)

        Returns:
            Dict: A dictionary where keys are file extensions (e.g., "txt", "jpg") and values are lists of items from file_input that correspond to that extension
        """
        classified = {"image_url": [], "video_url": [], "non_type": []}

        for file_path in file_input:
            ext = ""
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                ext = f".{ext}"

            if config._is_image_file(ext):
                classified["image_url"].append(file_path)
            elif config._is_video_file(ext):
                classified["video_url"].append(file_path)
            else:
                classified["non_type"].append(file_path)

        return classified

    def enable_debug(self, session_id: Optional[str] = None):
        """
        Enable debug mode for a specific session or globally

        Args:
            self: Instance reference
            session_id (Optional[str]): Optional session ID to enable debug for,
                if None, enables debug mode globally
        """
        self.debug_logger.enable_debug(session_id)

    def disable_debug(self):
        """
        Disable debug mode globally

        Turns off debug logging for all sessions
        """
        self.debug_logger.disable_debug()

    def get_debug_logs(self, session_id: Optional[str] = None) -> List[DebugInfo]:
        """
        Retrieve debug logs for a specific session or all sessions

        Args:
            self: Instance reference
            session_id (Optional[str]): Optional session ID to get logs for,
                if None, returns logs for all sessions

        Returns:
            List[DebugInfo]: List of debug information objects containing session logs
        """
        return self.debug_logger.get_session_logs(session_id)

    def export_debug_logs(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """
        Export debug logs in the specified format

        Args:
            self: Instance reference
            session_id (Optional[str]): Optional session ID to export logs for,
                if None, exports logs for all sessions
            format (str): Format to export logs in, defaults to "json"

        Returns:
            str: Exported logs as a string in the specified format
        """
        return self.debug_logger.export_session_logs(session_id, format)

    def clear_debug_logs(self, session_id: Optional[str] = None):
        """
        Clear debug logs for a specific session or all sessions

        Args:
            self: Instance reference
            session_id (Optional[str]): Optional session ID to clear logs for,
                if None, clears logs for all sessions
        """
        self.debug_logger.clear_session_logs(session_id)

    def stop(self):
        """
        Stop all running response generators

        Terminates processing for all active generators and stops any ongoing operations
        """
        for generator in self.generators.values():
            generator.stop()

    def reset(self):
        """
        Reset all response generators to their initial state

        Clears any accumulated state or context in generators, preparing them for new sessions
        """
        for generator in self.generators.values():
            generator.reset()

    def _determine_model_type(
        self, model_name: str, enable_thought: bool = False
    ) -> str:
        """
        Determine model type based on model name and configuration

        Analyzes the model name and thought process configuration to classify
        the model into an appropriate type category (e.g., text, multimodal, etc.)

        Args:
            self: Instance reference
            model_name (str): Name or identifier of the model to classify
            enable_thought (bool): Flag indicating if thought process generation is enabled,
                defaults to False

        Returns:
            str: String representing the determined model type
        """
        is_multimodal = config.is_vl_models(model_name)
        is_thought_capable = (
            True if enable_thought else config.is_thought_model(model_name)
        )

        if is_multimodal and is_thought_capable:
            return ModelType.MULTIMODAL_THOUGHT
        elif is_multimodal:
            return ModelType.MULTIMODAL
        elif is_thought_capable:
            return ModelType.THOUGHT
        else:
            return ModelType.TEXT

    # Legacy compatibility methods
    async def text_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,
        url_input: str = None,
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate text-only response (legacy compatibility)

        Args:
            message: User message (can contain image/video URLs for multimodal models)
            history: Conversation history
            model_name: Name of the model to use
            enable_thought: Whether to enable thought process
            role_setting: Role configuration
            system_prompt: System prompt
            max_length: Maximum response length
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            port: Service port
            file_input: List of file paths for multimodal content

        """
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            file_input=file_input,
            url_input=url_input,
        )

        generator = self.generators[ModelType.TEXT]
        async for result in generator.generate_response(request, self):
            yield result

    async def multimodal_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,
        url_input: str = None,
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate multimodal response (legacy compatibility)

        Args:
            message: User message (can contain image/video URLs for multimodal models)
            history: Conversation history
            model_name: Name of the model to use
            enable_thought: Whether to enable thought process
            role_setting: Role configuration
            system_prompt: System prompt
            max_length: Maximum response length
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            port: Service port
            file_input: List of file paths for multimodal content
        """
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            file_input=file_input,
            url_input=url_input,
        )

        generator = self.generators[ModelType.MULTIMODAL]
        async for result in generator.generate_response(request, self):
            yield result

    async def thought_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,
        url_input: str = None,
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Generate response with thought process (legacy compatibility)

        Args:
            message: User message (can contain image/video URLs for multimodal models)
            history: Conversation history
            model_name: Name of the model to use
            enable_thought: Whether to enable thought process
            role_setting: Role configuration
            system_prompt: System prompt
            max_length: Maximum response length
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            port: Service port
            file_input: List of file paths for multimodal content
        """
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            enable_thinking=True,
            file_input=file_input,
            url_input=url_input,
        )

        generator = self.generators[ModelType.THOUGHT]
        async for result in generator.generate_response(request, self):
            yield result

    async def generate_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        model_name: str,
        enable_thought: bool = False,
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,
        url_input: str = None,
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Unified interface to generate response based on model capabilities

        Args:
            message: User message (can contain image/video URLs for multimodal models)
            history: Conversation history
            model_name: Name of the model to use
            enable_thought: Whether to enable thought process
            role_setting: Role configuration
            system_prompt: System prompt
            max_length: Maximum response length
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            port: Service port
            file_input: List of file paths for multimodal content
        """
        request = ChatRequest(
            message=message,
            model_name=model_name,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            enable_thinking=enable_thought,
            file_input=file_input,
            url_input=url_input,
        )

        model_type = self._determine_model_type(model_name, enable_thought)
        generator = self.generators[model_type]

        async for result in generator.generate_response(request, self):
            yield result


chatbot = ChatBotGenerator(debug_enabled=True, debug_session_id="default_session")

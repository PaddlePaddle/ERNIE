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

"""
This script provides a Gradio interface for interacting with a chatbot based on Retrieval-Augmented Generation.
"""

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import textwrap
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from functools import partial

import faiss
import gradio as gr
import numpy as np
from bot_requests import BotClient

os.environ["NO_PROXY"] = "localhost,127.0.0.1"  # Disable proxy

logging.root.setLevel(logging.INFO)

FILE_URL_DEFAULT = "data/coffee.txt"
RELEVANT_PASSAGE_DEFAULT = textwrap.dedent(
    """\
    1675年时，英格兰就有3000多家咖啡馆；启蒙运动时期，咖啡馆成为民众深入讨论宗教和政治的聚集地，
    1670年代的英国国王查理二世就曾试图取缔咖啡馆。这一时期的英国人认为咖啡具有药用价值，
    甚至名医也会推荐将咖啡用于医疗。"""
)

QUERY_REWRITE_PROMPT = textwrap.dedent(
    """\
    【当前时间】
    {TIMESTAMP}

    【对话内容】
    {CONVERSATION}

    你的任务是根据上面user与assistant的对话内容，理解user意图，改写user的最后一轮对话，以便更高效地从知识库查找相关知识。具体的改写要求如下：
    1. 如果user的问题包括几个小问题，请将它们分成多个单独的问题。
    2. 如果user的问题涉及到之前对话的信息，请将这些信息融入问题中，形成一个不需要上下文就可以理解的完整问题。
    3. 如果user的问题是在比较或关联多个事物时，先将其拆分为单个事物的问题，例如‘A与B比起来怎么样’，拆分为：‘A怎么样’以及‘B怎么样’。
    4. 如果user的问题中描述事物的限定词有多个，请将多个限定词拆分成单个限定词。
    5. 如果user的问题具有**时效性（需要包含当前时间信息，才能得到正确的回复）**的时候，需要将当前时间信息添加到改写的query中；否则不加入当前时间信息。
    6. 只在**确有必要**的情况下改写，不需要改写时query输出[]。输出不超过 5 个改写问题，不要为了凑满数量而输出冗余问题。

    【输出格式】只输出 JSON ，不要给出多余内容
    ```json
    {{
    "query": ["改写问题1", "改写问题2"...]
    }}```
    """
)
ANSWER_PROMPT = textwrap.dedent(
    """\
    你是阅读理解问答专家。

    【文档知识】
    {DOC_CONTENT}

    你的任务是根据对话内容，理解用户需求，参考文档知识回答用户问题，知识参考详细原则如下：
    - 对于同一信息点，如文档知识与模型通用知识均可支撑，应优先以文档知识为主，并对信息进行验证和综合。
    - 如果文档知识不足或信息冲突，必须指出“根据资料无法确定”或“不同资料存在矛盾”，不得引入文档知识与通识之外的主观推测。

    同时，回答问题需要综合考虑规则要求中的各项内容，详细要求如下：
    【规则要求】
    * 回答问题时，应优先参考与问题紧密相关的文档知识，不要在答案中引入任何与问题无关的文档内容。
    * 回答中不可以让用户知道你查询了相关文档。
    * 回复答案不要出现'根据文档知识'，'根据当前时间'等表述。
    * 论述突出重点内容，以分点条理清晰的结构化格式输出。

    【当前时间】
    {TIMESTAMP}

    【对话内容】
    {CONVERSATION}

    直接输出回复内容即可。
    """
)
QUERY_DEFAULT = "1675 年时，英格兰有多少家咖啡馆？"


def get_args() -> argparse.Namespace:
    """
    Parse and return command line arguments for the ERNIE models chat demo.
    Configures server settings, model endpoint, and document processing parameters.

    Returns:
        argparse.Namespace: Parsed command line arguments containing all the above settings.
    """
    parser = ArgumentParser(description="ERNIE models web chat demo.")

    parser.add_argument(
        "--server-port", type=int, default=8333, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--max_char",
        type=int,
        default=20000,
        help="Maximum character limit for messages.",
    )
    parser.add_argument(
        "--max_retry_num", type=int, default=3, help="Maximum retry number for request."
    )
    parser.add_argument(
        "--model_map",
        required=True,
        type=str,
        default='{"ERNIE-4.5": "http://localhost:port/v1"}',
        help="""JSON string defining model name to endpoint mappings.
            Required Format:
            {"ERNIE-4.5": "http://localhost:port/v1"}

            Note:
            - Endpoints must be valid HTTP URL
            - Specify ONE model endpoint in JSON format.
            - Prefix determines model capabilities:
            * ERNIE-4.5: Text-only model
            """,
    )
    parser.add_argument(
        "--embedding_service_url",
        type=str,
        default="https://qianfan.baidubce.com/v2",
        help="Embedding service url.",
    )
    parser.add_argument(
        "--qianfan_api_key",
        type=str,
        default="bce-v3/xxx",
        help="Qianfan API key.",
        required=True,
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="embedding-v1",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=384,
        help="Dimension of the embedding vector.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for splitting long documents.",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Top k results to retrieve."
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default="data/faiss_index",
        help="Faiss index path.",
    )
    parser.add_argument(
        "--text_db_path",
        type=str,
        default="data/text_db.jsonl",
        help="Text database path.",
    )
    parser.add_argument(
        "--concurrency_limit", type=int, default=10, help="Default concurrency limit."
    )
    parser.add_argument(
        "--max_queue_size", type=int, default=50, help="Maximum queue size for request."
    )

    args = parser.parse_args()
    try:
        args.model_map = json.loads(args.model_map)

        # Validation: Check at least one model exists
        if len(args.model_map) < 1:
            raise ValueError("model_map must contain at least one model configuration")
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for --model_map") from e

    return args


class FaissTextDatabase:
    """
    A vector database for text retrieval using FAISS.
    Provides efficient similarity search and document management capabilities.
    """

    def __init__(self, args, bot_client: BotClient):
        """
        Initialize the FaissTextDatabase.

        Args:
            args: arguments for initialization
            bot_client: instance of BotClient
            embedding_dim: dimension of the embedding vector
        """
        self.logger = logging.getLogger(__name__)

        self.bot_client = bot_client
        self.embedding_dim = getattr(args, "embedding_dim", 384)
        self.top_k = getattr(args, "top_k", 3)
        self.context_size = getattr(args, "context_size", 2)
        self.faiss_index_path = getattr(args, "faiss_index_path", "data/faiss_index")
        self.text_db_path = getattr(args, "text_db_path", "data/text_db.jsonl")

        # If faiss_index_path exists, load it and text_db_path
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.text_db_path):
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.text_db_path, "r", encoding="utf-8") as f:
                self.text_db = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.text_db = {
                "file_md5s": [],
                "chunks": [],
            }  # Save file_md5s to avoid duplicates  # Save chunks

    def calculate_md5(self, file_path: str) -> str:
        """
        Calculate the MD5 hash of a file

        Args:
            file_path: the path of the source file

        Returns:
            str: the MD5 hash
        """
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def is_file_processed(self, file_path: str) -> bool:
        """
        Check if the file has been processed before

        Args:
            file_path: the path of the source file

        Returns:
            bool: whether the file has been processed
        """
        file_md5 = self.calculate_md5(file_path)
        return file_md5 in self.text_db["file_md5s"]

    def add_embeddings(
        self,
        file_path: str,
        segments: list[str],
        progress_bar: gr.Progress = None,
        save_file: bool = False,
    ) -> bool:
        """
        Stores document embeddings in FAISS database after checking for duplicates.
        Generates embeddings for each text segment, updates the FAISS index and metadata database,
        and persists changes to disk. Includes optional progress tracking for Gradio interfaces.

        Args:
            file_path: the path of the source file
            segments: the list of segments
            progress_bar: the progress bar object

        Returns:
            bool: whether the operation was successful
        """
        file_md5 = self.calculate_md5(file_path)
        if file_md5 in self.text_db["file_md5s"]:
            self.logger.info(f"File already processed: {file_path} (MD5: {file_md5})")
            return False

        # Generate embeddings
        vectors = []
        file_name = os.path.basename(file_path)
        file_txt = "".join(file_name.split(".")[:-1])[:30]
        for i, segment in enumerate(segments):
            vectors.append(self.bot_client.embed_fn(file_txt + "\n" + segment))
            if progress_bar is not None:
                progress_bar((i + 1) / len(segments), desc=file_name + " Processing...")
        vectors = np.array(vectors)
        self.index.add(vectors.astype("float32"))

        start_id = len(self.text_db["chunks"])
        for i, text in enumerate(segments):
            self.text_db["chunks"].append(
                {
                    "file_md5": file_md5,
                    "file_name": file_name,
                    "file_txt": file_txt,
                    "text": text,
                    "vector_id": start_id + i,
                }
            )

        self.text_db["file_md5s"].append(file_md5)
        if save_file:
            self.save()
        return True

    def search_with_context(self, query_list: list) -> str:
        """
        Finds the most relevant text chunks for multiple queries and includes surrounding context.
        Uses FAISS to find the closest matching embeddings, then retrieves adjacent chunks
        from the same source document to provide better context understanding.

        Args:
            query_list: list of input query strings

        Returns:
            str: the concatenated output string
        """
        # Step 1: Retrieve top_k results for each query and collect all indices
        all_indices = []
        for query in query_list:
            query_vector = np.array([self.bot_client.embed_fn(query)]).astype("float32")
            _, indices = self.index.search(query_vector, self.top_k)
            all_indices.extend(indices[0].tolist())

        # Step 2: Remove duplicate indices
        unique_indices = sorted(set(all_indices))
        self.logger.info(f"Retrieved indices: {all_indices}")
        self.logger.info(f"Unique indices after deduplication: {unique_indices}")

        # Step 3: Expand each index with context (within same file boundaries)
        expanded_indices = set()
        file_boundaries = {}  # {file_md5: (start_idx, end_idx)}
        for target_idx in unique_indices:
            target_chunk = self.text_db["chunks"][target_idx]
            target_file_md5 = target_chunk["file_md5"]

            if target_file_md5 not in file_boundaries:
                file_start = target_idx
                while (
                    file_start > 0
                    and self.text_db["chunks"][file_start - 1]["file_md5"]
                    == target_file_md5
                ):
                    file_start -= 1
                file_end = target_idx
                while (
                    file_end < len(self.text_db["chunks"]) - 1
                    and self.text_db["chunks"][file_end + 1]["file_md5"]
                    == target_file_md5
                ):
                    file_end += 1
            else:
                file_start, file_end = file_boundaries[target_file_md5]

            # Calculate context range within file boundaries
            start = max(file_start, target_idx - self.context_size)
            end = min(file_end, target_idx + self.context_size)

            for pos in range(start, end + 1):
                expanded_indices.add(pos)

        # Step 4: Sort and merge continuous chunks
        sorted_indices = sorted(expanded_indices)
        groups = []
        current_group = [sorted_indices[0]]
        for i in range(1, len(sorted_indices)):
            if (
                sorted_indices[i] == sorted_indices[i - 1] + 1
                and self.text_db["chunks"][sorted_indices[i]]["file_md5"]
                == self.text_db["chunks"][sorted_indices[i - 1]]["file_md5"]
            ):
                current_group.append(sorted_indices[i])
            else:
                groups.append(current_group)
                current_group = [sorted_indices[i]]
        groups.append(current_group)

        # Step 5: Create merged text for each group
        result = ""
        for idx, group in enumerate(groups):
            result += "\n段落{idx}:\n{title}\n".format(
                idx=idx + 1, title=self.text_db["chunks"][group[0]]["file_txt"]
            )
            for idx in group:
                result += self.text_db["chunks"][idx]["text"] + "\n"
            self.logger.info(f"Merged chunk range: {group[0]}-{group[-1]}")

        return result

    def save(self) -> None:
        """Save the database to disk."""
        faiss.write_index(self.index, self.faiss_index_path)

        with open(self.text_db_path, "w", encoding="utf-8") as f:
            json.dump(self.text_db, f, ensure_ascii=False, indent=2)


class GradioEvents:
    """
    Manages event handling and UI interactions for Gradio applications.
    Provides methods to process user inputs, trigger callbacks, and update interface components.
    """

    @staticmethod
    def get_history_conversation(task_history: list) -> tuple:
        """
        Converts task history into conversation format for model processing.
        Transforms query-response pairs into structured message history and plain text.

        Args:
            task_history (list): List of tuples containing queries and responses.

        Returns:
            tuple: Tuple containing two elements:
                - conversation (list): List of dictionaries representing the conversation history.
                - conversation_str (str): String representation of the conversation history.
        """
        conversation = []
        conversation_str = ""
        for query_h, response_h in task_history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
            conversation_str += f"user:\n{query_h}\n assistant:\n{response_h}\n "
        return conversation, conversation_str

    @staticmethod
    def chat_stream(
        query: str,
        task_history: list,
        model: str,
        faiss_db: FaissTextDatabase,
        bot_client: BotClient,
    ) -> dict:
        """
        Streams chatbot responses by processing queries with context from history and FAISS database.
        Integrates language model generation with knowledge retrieval to produce dynamic responses.
        Yields response events in real-time for interactive conversation experiences.

        Args:
            query (str): The query string.
            task_history (list): The task history record list.
            model (Model): The model used to generate responses.
            bot_client (BotClient): The chatbot client object.
            faiss_db (FaissTextDatabase): The FAISS database object.

        Yields:
            dict: A dictionary containing the event type and its corresponding content.
        """
        conversation, conversation_str = GradioEvents.get_history_conversation(
            task_history
        )
        conversation_str += f"user:\n{query}\n"

        search_info_message = QUERY_REWRITE_PROMPT.format(
            TIMESTAMP=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            CONVERSATION=conversation_str,
        )
        search_conversation = [{"role": "user", "content": search_info_message}]
        search_info_result = GradioEvents.get_sub_query(
            search_conversation, model, bot_client
        )
        if search_info_result is None:
            search_info_result = {"query": [query]}

        if search_info_result.get("query", []):
            relevant_passages = faiss_db.search_with_context(
                search_info_result["query"]
            )
            yield {"type": "relevant_passage", "content": relevant_passages}

            query = ANSWER_PROMPT.format(
                DOC_CONTENT=relevant_passages,
                TIMESTAMP=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                CONVERSATION=conversation_str,
            )

        conversation.append({"role": "user", "content": query})
        try:
            req_data = {"messages": conversation}
            for chunk in bot_client.process_stream(model, req_data):
                if "error" in chunk:
                    raise Exception(chunk["error"])

                message = chunk.get("choices", [{}])[0].get("delta", {})
                content = message.get("content", "")

                if content:
                    yield {"type": "answer", "content": content}

        except Exception as e:
            raise gr.Error("Exception: " + repr(e))

    @staticmethod
    def predict_stream(
        query: str,
        chatbot: list,
        task_history: list,
        model: str,
        faiss_db: FaissTextDatabase,
        bot_client: BotClient,
    ) -> tuple:
        """
        Generates streaming responses by combining model predictions with knowledge retrieval.
        Processes user queries using conversation history and FAISS database context,
        yielding updated chat messages and relevant passages in real-time.

        Args:
            query (str): The content of the user's input query.
            chatbot (list): The chatbot's historical message list.
            task_history (list): The task history record list.
            model (Model): The model used to generate responses.
            bot_client (object): The chatbot client object.
            faiss_db (FaissTextDatabase): The FAISS database instance.

        Yields:
            tuple: A tuple containing the updated chatbot's message list and the relevant passage.
        """
        query = query if query else QUERY_DEFAULT

        logging.info(f"User: {query}")
        chatbot.append({"role": "user", "content": query})

        # First yield the chatbot with user message
        yield chatbot, None

        new_texts = GradioEvents.chat_stream(
            query,
            task_history,
            model,
            faiss_db,
            bot_client,
        )

        response = ""
        current_relevant_passage = None
        for new_text in new_texts:
            if not isinstance(new_text, dict):
                continue

            if new_text.get("type") == "embedding":
                current_relevant_passage = new_text["content"]
                yield chatbot, current_relevant_passage
                continue
            elif new_text.get("type") == "relevant_passage":
                current_relevant_passage = new_text["content"]
                yield chatbot, current_relevant_passage
                continue
            elif new_text.get("type") == "answer":
                response += new_text["content"]

            # Remove previous message if exists
            if chatbot[-1].get("role") == "assistant":
                chatbot.pop(-1)

            if response:
                chatbot.append({"role": "assistant", "content": response})
                yield chatbot, current_relevant_passage

        logging.info(f"History: {task_history}")
        task_history.append((query, response))
        logging.info(f"ERNIE models: {response}")

    @staticmethod
    def regenerate(
        chatbot: list,
        task_history: list,
        model: str,
        faiss_db: FaissTextDatabase,
        bot_client: BotClient,
    ) -> tuple:
        """
        Regenerate the chatbot's response based on the latest user query

        Args:
            chatbot (list): Chat history list
            task_history (list): Task history
            model (str): Model name to use
            bot_client (BotClient): Bot request client instance
            faiss_db (FaissTextDatabase): Faiss database instance

        Yields:
            tuple: Updated chatbot and relevant_passage
        """
        if not task_history:
            yield chatbot, None
            return
        # Pop the last user query and bot response from task_history
        item = task_history.pop(-1)
        while len(chatbot) != 0 and chatbot[-1].get("role") == "assistant":
            chatbot.pop(-1)
        chatbot.pop(-1)

        yield from GradioEvents.predict_stream(
            item[0],
            chatbot,
            task_history,
            model,
            faiss_db,
            bot_client,
        )

    @staticmethod
    def reset_user_input() -> gr.update:
        """
        Reset user input box content.

        Returns:
            gr.update: An update object representing the cleared value
        """
        return gr.update(value="")

    @staticmethod
    def reset_state() -> namedtuple:
        """
        Reset chat state and clear all history.

        Returns:
            tuple: A named tuple containing the updated values for chatbot, task_history, file_btn, and relevant_passage
        """
        GradioEvents.gc()

        reset_result = namedtuple(
            "reset_result", ["chatbot", "task_history", "file_btn", "relevant_passage"]
        )
        return reset_result(
            [],  # clear chatbot
            [],  # clear task_history
            gr.update(value=None),  # clear file_btn
            gr.update(value=None),  # reset relevant_passage
        )

    @staticmethod
    def gc():
        """
        Force garbage collection to free memory.
        """
        import gc

        gc.collect()

    @staticmethod
    def get_image_url(image_path: str) -> str:
        """
        Encode image file to Base64 format and generate data URL.
        Reads an image file from disk, encodes it as Base64, and formats it
        as a data URL that can be used directly in HTML or API requests.

        Args:
            image_path (str): Path to the image file. Must be a valid file path.

        Returns:
            str: Data URL string in format "data:image/{ext};base64,{encoded_data}"
        """
        base64_image = ""
        extension = image_path.split(".")[-1]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/{extension};base64,{base64_image}"
        return url

    @staticmethod
    def get_sub_query(
        conversation: list, model_name: str, bot_client: BotClient
    ) -> dict:
        """
        Enhances user queries by generating alternative phrasings using language models.
        Creates semantically similar variations of the original query to improve retrieval accuracy.
        Returns structured dictionary containing both original and rephrased queries.

        Args:
            conversation (list): The conversation history.
            model_name (str): The name of the model to use for rephrasing.
            bot_client (BotClient): The bot client instance.

        Returns:
            dict: The rephrased query.
        """
        req_data = {"messages": conversation}
        try:
            response = bot_client.process(model_name, req_data)
            search_info_res = response["choices"][0]["message"]["content"]
            start = search_info_res.find("{")
            end = search_info_res.rfind("}") + 1
            if start >= 0 and end > start:
                search_info_res = search_info_res[start:end]
            search_info_res = json.loads(search_info_res)
            if search_info_res.get("sub_query_list", []):
                unique_list = list(set(search_info_res["sub_query_list"]))
                search_info_res["sub_query_list"] = unique_list
            return search_info_res
        except Exception:
            logging.error("Error: Model output is not a valid JSON")
            return None

    @staticmethod
    def split_oversized_line(line: str, chunk_size: int) -> tuple:
        """
        Split a line into two parts based on punctuation marks or whitespace while preserving
        natural language boundaries and maintaining the original content structure.

        Args:
            line (str): The line to split.
            chunk_size (int): The maximum length of each chunk.

        Returns:
            tuple: Two strings, the first part of the original line and the rest of the line.
        """
        PUNCTUATIONS = {
            ".",
            "。",
            "!",
            "！",
            "?",
            "？",
            ",",
            "，",
            ";",
            "；",
            ":",
            "：",
        }

        if len(line) <= chunk_size:
            return line, ""

        # Search from chunk_size position backwards
        split_pos = chunk_size
        for i in range(chunk_size, 0, -1):
            if line[i] in PUNCTUATIONS:
                split_pos = i + 1  # Include punctuation
                break

        # Fallback to whitespace if no punctuation found
        if split_pos == chunk_size:
            split_pos = line.rfind(" ", 0, chunk_size)
            if split_pos == -1:
                split_pos = chunk_size  # Hard split

        return line[:split_pos], line[split_pos:]

    @staticmethod
    def split_text_into_chunks(file_url: str, chunk_size: int) -> list:
        """
        Split file text into chunks of a specified size while respecting natural language boundaries
        and avoiding mid-word splits whenever possible.

        Args:
            file_url (str): The file URL.
            chunk_size (int): The maximum length of each chunk.

        Returns:
            list: A list of strings, where each element represents a chunk of the original text.
        """
        with open(file_url, "r", encoding="utf-8") as f:
            text = f.read()

        if not text:
            logging.error("Error: File is empty")
            return []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for line in lines:
            # If adding this line would exceed chunk size (and we have content)
            if current_length + len(line) > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Process oversized lines first
            while len(line) > chunk_size:
                head, line = GradioEvents.split_oversized_line(line, chunk_size)
                chunks.append(head)

            # Add remaining line content
            if line:
                current_chunk.append(line)
                current_length += len(line) + 1

        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    @staticmethod
    def file_upload(
        files_url: list,
        chunk_size: int,
        faiss_db: FaissTextDatabase,
        progress_bar: gr.Progress = gr.Progress(),
    ) -> str:
        """
        Uploads and processes multiple files by splitting them into semantically meaningful chunks,
        then indexes them in the FAISS database with progress tracking.

        Args:
            files_url (list): List of file URLs.
            chunk_size (int): Maximum chunk size.
            faiss_db (FaissTextDatabase): FAISS database instance.
            progress_bar (gr.Progress): Progress bar instance.

        Returns:
            str: Message indicating successful completion.
        """
        if not files_url:
            return
        yield gr.update(visible=True)
        for file_url in files_url:
            if not GradioEvents.save_file_to_db(
                file_url, chunk_size, faiss_db, progress_bar
            ):
                file_name = os.path.basename(file_url)
                gr.Info(f"{file_name} already processed.")

        yield gr.update(visible=False)

    @staticmethod
    def save_file_to_db(
        file_url: str,
        chunk_size: int,
        faiss_db: FaissTextDatabase,
        progress_bar: gr.Progress = None,
        save_file: bool = False,
    ):
        """
        Processes and indexes document content into FAISS database with semantic-aware chunking.
        Handles file validation, text segmentation, embedding generation and storage operations.

        Args:
            file_url (str): File URL.
            chunk_size (int): Chunk size.
            faiss_db (FaissTextDatabase): FAISS database instance.
            progress_bar (gr.Progress): Progress bar instance.

        Returns:
            bool: True if the file was saved successfully, otherwise False.
        """
        if not os.path.exists(file_url):
            logging.error(f"File not found: {file_url}")
            return False

        file_name = os.path.basename(file_url)
        if not faiss_db.is_file_processed(file_url):
            logging.info(f"{file_url} not processed yet, processing now...")
            try:
                segments = GradioEvents.split_text_into_chunks(file_url, chunk_size)
                faiss_db.add_embeddings(file_url, segments, progress_bar, save_file)

                logging.info(f"{file_url} processed successfully.")
                return True
            except Exception as e:
                logging.error(f"Error processing {file_url}: {e!s}")
                gr.Error(f"Error processing file: {file_name}")
                raise
        else:
            logging.info(f"{file_url} already processed.")
            return False


def launch_demo(
    args: argparse.Namespace,
    bot_client: BotClient,
    faiss_db_template: FaissTextDatabase,
):
    """
    Launch demo program

    Args:
        args (argparse.Namespace): argparse Namespace object containing parsed command line arguments
        bot_client (BotClient): Bot client instance
        faiss_db (FaissTextDatabase): FAISS database instance
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
        font-size: 18px;
        color: #555;
        margin-top: 8px;
        white-space: nowrap;
    }
    """
    with gr.Blocks(css=css) as demo:
        model_name = gr.State(next(iter(args.model_map.keys())))
        faiss_db = gr.State(copy.deepcopy(faiss_db_template))

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
        gr.Markdown(
            """\
<center><font size=3>    <a href="https://ernie.baidu.com/">ERNIE Bot</a> | \
<a href="https://github.com/PaddlePaddle/ERNIE">GitHub</a> | \
<a href="https://huggingface.co/baidu">Hugging Face</a> | \
<a href="https://aistudio.baidu.com/modelsoverview">BAIDU AI Studio</a> | \
<a href="https://yiyan.baidu.com/blog/publication/">Technical Report</a></center>"""
        )

        chatbot = gr.Chatbot(label="ERNIE", type="messages")

        with gr.Row(equal_height=True):
            file_btn = gr.File(
                label="Knowledge Base Upload (System default will be used if none provided. Accepted formats: TXT, MD)",
                height="150px",
                file_types=[".txt", ".md"],
                elem_id="file-upload",
                file_count="multiple",
            )
            relevant_passage = gr.Textbox(
                label="Relevant Passage",
                lines=5,
                max_lines=5,
                placeholder=RELEVANT_PASSAGE_DEFAULT,
                interactive=False,
            )
        with gr.Row():
            progress_bar = gr.Textbox(label="Progress", visible=False)

        query = gr.Textbox(label="Query", elem_id="text_input", value=QUERY_DEFAULT)

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History(清除历史)")
            submit_btn = gr.Button("🚀 Submit(发送)", elem_id="submit-button")
            regen_btn = gr.Button("🤔️ Regenerate(重试)")

        task_history = gr.State([])

        predict_with_clients = partial(
            GradioEvents.predict_stream, bot_client=bot_client
        )
        regenerate_with_clients = partial(
            GradioEvents.regenerate, bot_client=bot_client
        )
        file_upload_with_clients = partial(
            GradioEvents.file_upload,
        )

        chunk_size = gr.State(args.chunk_size)
        file_btn.change(
            fn=file_upload_with_clients,
            inputs=[file_btn, chunk_size, faiss_db],
            outputs=[progress_bar],
        )
        query.submit(
            predict_with_clients,
            inputs=[query, chatbot, task_history, model_name, faiss_db],
            outputs=[chatbot, relevant_passage],
            show_progress=True,
        )
        query.submit(GradioEvents.reset_user_input, [], [query])
        submit_btn.click(
            predict_with_clients,
            inputs=[query, chatbot, task_history, model_name, faiss_db],
            outputs=[chatbot, relevant_passage],
            show_progress=True,
        )
        submit_btn.click(GradioEvents.reset_user_input, [], [query])
        empty_btn.click(
            GradioEvents.reset_state,
            outputs=[chatbot, task_history, file_btn, relevant_passage],
            show_progress=True,
        )
        regen_btn.click(
            regenerate_with_clients,
            inputs=[chatbot, task_history, model_name, faiss_db],
            outputs=[chatbot, relevant_passage],
            show_progress=True,
        )

    demo.queue(
        default_concurrency_limit=args.concurrency_limit, max_size=args.max_queue_size
    )
    demo.launch(server_port=args.server_port, server_name=args.server_name)


def main():
    """Main function that runs when this script is executed."""
    args = get_args()
    bot_client = BotClient(args)
    faiss_db = FaissTextDatabase(args, bot_client)

    # Run file upload function to save default knowledge base.
    GradioEvents.save_file_to_db(
        FILE_URL_DEFAULT, args.chunk_size, faiss_db, save_file=True
    )

    launch_demo(args, bot_client, faiss_db)


if __name__ == "__main__":
    main()

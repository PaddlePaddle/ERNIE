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
Process execution management, initiation, and supervision
"""

import asyncio
import os
import re
import subprocess

from erniekit.webui import common
from erniekit.webui.alert import alert


class CommandRunner:

    def __init__(self):
        self.current_process = None
        self.process_lock = asyncio.Lock()
        self.was_terminated_by_user = False
        self.lines_history = []
        self.track_progress = True
        self.current = 0
        self.total = 0
        self.progress_line_buffer = {}

    async def execute(self, command: str):
        """
        Asynchronously execute a shell command and stream its output.

        Args:
            self: Instance reference
            command (str):  command to execute

        Returns:
            AsyncGenerator[str, None]: Asynchronous generator yielding output lines
        """
        self.lines_history = []
        self.progress_line_buffer = {}
        separator = "\n" + "-" * 50 + "\n"
        start_text = alert.get("progress", "run_command").format(separator, command) + "\n"
        self.lines_history.extend([start_text])

        yield "\n".join(self.lines_history), 0, 0

        print("\n" + start_text, flush=True)
        process = None

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FORCE_COLOR"] = "1"
            process = await asyncio.create_subprocess_shell(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            )
            self.current_process = process

            buffer = b""
            while True:
                chunk = await process.stdout.read(1024)
                if not chunk:
                    break

                buffer += chunk
                while b"\n" in buffer or b"\r" in buffer:
                    line, buffer = self._extract_next_line(buffer)
                    if not line:
                        break

                    line_str = line.decode("utf-8", errors="replace")

                    print(line_str, end="", flush=True)

                    line_clean = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", line_str)
                    line_clean = line_clean.rstrip("\n\r").strip()

                    if line_clean:
                        should_update = self._process_line(line_clean)
                        self._parse_progress(line_clean)

                        if should_update:
                            yield "\n".join(self.lines_history), self.current, self.total

        except Exception as e:
            error_msg = alert.get("progress", "execution_error").format(str(e))
            self.lines_history.append(error_msg)
            print(error_msg, flush=True)
            yield "\n".join(self.lines_history), self.current, self.total

        finally:
            self._flush_progress_buffer()

            if process:
                return_code = await process.wait()
                if return_code == 0:
                    success_msg = alert.get("progress", "progress_success")
                    self.lines_history.append(f"\n{success_msg}")
                    print(f"\n{success_msg}", flush=True)
                    yield "\n".join(self.lines_history), self.current, self.total
            self.current_process = None

    def _extract_next_line(self, buffer):
        """
        Extract a complete line from the buffer, handling both LF (\n) and CR+LF (\r\n) endings.

        Args:
            self: Instance reference
            buffer (bytes): Byte buffer containing partial or complete lines
        """
        nl_pos = buffer.find(b"\n")
        cr_pos = buffer.find(b"\r")

        if nl_pos >= 0 and cr_pos >= 0:
            end_pos = min(nl_pos, cr_pos) + 1
        elif nl_pos >= 0:
            end_pos = nl_pos + 1
        elif cr_pos >= 0:
            end_pos = cr_pos + 1
        else:
            return buffer, b""

        return buffer[:end_pos], buffer[end_pos:]

    def _process_line(self, line_clean):
        """
        Process a single line of output to determine if frontend updates are required.

        Args:
            self: Instance reference
            line_clean (str): Cleaned output line without line endings

        Returns:
            bool: True if frontend update is needed, False otherwise
        """
        progress_key = self._get_progress_key(line_clean)
        if progress_key:
            self.progress_line_buffer[progress_key] = line_clean

            if self._should_show_progress(line_clean):
                self._update_progress_in_history(progress_key, line_clean)
                return True
            return False
        else:
            self.lines_history.append(line_clean)
            return True

    def _get_progress_key(self, line):
        """
        Extract a key from the line to identify the associated progress bar.

        Args:
            self: Instance reference
            line (str): Line of text potentially containing progress information

        Returns:
            Optional[str]: Progress bar identifier if found, None otherwise
        """
        patterns = [
            r"(Loading\s+\w+):\s*\d+%",
            r"(\w+\s+\w+):\s*\d+%",
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

        return None

    def _should_show_progress(self, line):
        """
        Determine if a progress line should be displayed based on predefined criteria.

        Args:
            self: Instance reference
            line (str): Line of text potentially containing progress information

        Returns:
            bool: True if the progress should be shown, False otherwise
        """
        percent_match = re.search(r"(\d+)%", line)
        if percent_match:
            percent = int(percent_match.group(1))
            return percent == 0 or percent == 100 or percent % 10 == 0

        return "100%" in line or "complete" in line.lower() or "finished" in line.lower()

    def _update_progress_in_history(self, progress_key, line):
        """
        Update or insert a progress bar line in the history buffer.

        Args:
            self: Instance reference
            progress_key (str): Unique identifier for the progress bar
            line (str): New progress line to update/insert
        """
        for i, history_line in enumerate(self.lines_history):
            if progress_key in history_line and self._get_progress_key(history_line):
                self.lines_history[i] = line
                return

        self.lines_history.append(line)

    def _flush_progress_buffer(self):
        """
        Flush all buffered progress updates to the history buffer.

        Args:
            self: Instance reference

        """
        for progress_key, line in self.progress_line_buffer.items():
            self._update_progress_in_history(progress_key, line)

    def _parse_progress(self, line):
        """
        Parse progress information (e.g., global_step, X/Y format) from a line of text.

        Args:
            self: Instance reference
            line (str): Line of text potentially containing progress data
        """
        try:
            step_match = re.search(r"global_step:\s*(\d+)", line)
            if step_match:
                step = int(step_match.group(1))
                self.current = step
                return

            ratio_match = re.search(r"(\d+)/(\d+)", line)
            if ratio_match:
                self.current = int(ratio_match.group(1))
                self.total = int(ratio_match.group(2))
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    async def stop(self):
        """
        Terminate the currently running process asynchronously.

        Args:
            self: Instance reference
        """
        async with self.process_lock:
            process = self.current_process

            if process is None:
                no_terminated_msg = "\n" + alert.get("progress", "no_progress") + "\n"
                self.lines_history.append(no_terminated_msg)
                return "\n".join(self.lines_history)

            try:
                if process.returncode is not None:
                    progress_end_msg = "\n" + alert.get("progress", "progress_end") + "\n"
                    self.lines_history.append(progress_end_msg)
                    return "\n".join(self.lines_history)

                try:
                    common.abort_process(process.pid)
                except Exception:
                    process.terminate()

                await asyncio.sleep(0.5)

                if process.returncode is None:
                    process.kill()
                    force_terminated_msg = "\n" + alert.get("progress", "force_terminated") + "\n"
                    print(force_terminated_msg)
                    self.lines_history.append(force_terminated_msg)
                    await process.wait()

                self.was_terminated_by_user = True
                user_terminated_msg = "\n" + alert.get("progress", "user_terminated") + "\n"
                self.lines_history.append(user_terminated_msg)
                print(user_terminated_msg)
            except Exception as e:
                error_msg = alert.get("progress", "terminate_error").format(str(e))
                self.lines_history.append(error_msg)
                print(error_msg.strip())
            finally:
                self.current_process = None

            return "\n".join(self.lines_history)

    def clear_output(self):
        """
        Clear the current output buffer.

        Args:
            self: Instance reference

        Returns:
            str: Empty string indicating successful clearing
        """
        self.output_reset = True
        self.current_output = ""
        self.progress_line_buffer = {}
        return ""

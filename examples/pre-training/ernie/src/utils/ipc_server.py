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
IPCServer
"""
import enum
import logging
from dataclasses import dataclass
from multiprocessing import Process, Queue, Lock


logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class ServerStatus(enum.Enum):
    """
    ServerStatus
    """

    WAIT_RUNNING = 0
    RUNNING = 1
    EXIT_WITH_FAILURE = 2
    EXIT_WITH_CLOSE = 3


class ResponseTag(enum.Enum):
    """
    ResponseTag
    """

    SUCCESS = 0
    FAILURE = 1


class ExitFlag:
    """
    ExitFlag
    """

    pass


@dataclass
class MethodRequest:
    """
    MethodRequest
    """

    router_key: object
    name: str
    args: list
    kwargs: dict


@dataclass
class AttrRequest:
    """
    AttrRequest
    """

    router_key: object
    name: str


@dataclass
class Response:
    """
    Response
    """

    tag: ResponseTag
    value: object
    exception: Exception


def server_loop(init_func, server_idx, server_num, init_queue, send_queue, recv_queue):
    """
    server_loop
    """
    try:
        init_obj = init_func(server_idx, server_num)
        init_queue.put(
            Response(
                tag=ResponseTag.SUCCESS, exception=None, value=ServerStatus.RUNNING
            )
        )
    except Exception as e:
        logger.exception(e)
        init_queue.put(
            Response(
                tag=ResponseTag.FAILURE,
                exception=e,
                value=ServerStatus.EXIT_WITH_FAILURE,
            )
        )
        return

    while True:
        request = send_queue.get()
        if isinstance(request, ExitFlag):
            break

        try:
            value = getattr(init_obj, request.name)
            if isinstance(request, MethodRequest):
                args = request.args or tuple()
                kwargs = request.kwargs or dict()
                value = value(*args, **kwargs)
            response = Response(tag=ResponseTag.SUCCESS, exception=None, value=value)
        except Exception as e:
            response = Response(tag=ResponseTag.FAILURE, exception=e, value=None)
            print("Exception inside process", e)

        recv_queue.put(response)


class SubIPCServer:
    """
    SubIPCServer
    """

    def __init__(self, server_idx, server_num, init_func):
        """
        __init__
        """
        self.send_queue = Queue()
        self.recv_queue = Queue()
        self.init_queue = Queue()
        self.server_status = ServerStatus.WAIT_RUNNING
        self.server_idx = server_idx
        self.server_num = server_num
        self.process = Process(
            target=server_loop,
            args=(
                init_func,
                server_idx,
                server_num,
                self.init_queue,
                self.send_queue,
                self.recv_queue,
            ),
        )
        self.process.daemon = True
        self.process.start()
        self.lock = Lock()

    def wait_started(self):
        """
        wait_started
        """
        if self.server_status == ServerStatus.RUNNING:
            return
        elif self.server_status == ServerStatus.WAIT_RUNNING:
            init_response = self.init_queue.get()
            assert init_response.value in [
                ServerStatus.RUNNING,
                ServerStatus.EXIT_WITH_FAILURE,
            ], init_response.value
            self.server_status = init_response.value
            if init_response.value == ServerStatus.EXIT_WITH_FAILURE:
                self.server_status = ServerStatus.EXIT_WITH_FAILURE
                raise init_response.exception
        elif self.server_status == ServerStatus.EXIT_WITH_FAILURE:
            raise RuntimeError("IPCServer does not start successfully")
        elif self.server_status == ServerStatus.EXIT_WITH_CLOSE:
            raise RuntimeError("IPCServer has been closed")
        else:
            raise RuntimeError(f"Unknown server status {self.server_status}")

    def response(self, request):
        """
        response
        """
        with self.lock:
            self.wait_started()
            self.send_queue.put(request)
            ret = self.recv_queue.get()
        return ret

    def close(self):
        """
        close
        """
        with self.lock:
            if self.process is not None:
                self.wait_started()
                self.send_queue.put(ExitFlag())
                self.process.join()
                self.process = None
                self.server_status = ServerStatus.EXIT_WITH_CLOSE


class IPCServer:
    """
    IPCServer
    """

    def __init__(self, router_groups, init_funcs):
        """
        __init__
        """
        server_num = len(init_funcs)
        group_num = len(router_groups)
        assert server_num == group_num, f"{server_num} vs {group_num}"
        assert (
            server_num > 0
        ), f"server_num should be larger than 0, but got {server_num}"
        self.router_map = {}
        self.sub_servers = [None] * server_num
        for i, (group, init_func) in enumerate(zip(router_groups, init_funcs)):
            sub_server = SubIPCServer(i, server_num, init_func)
            for router_key in group:
                if router_key in self.router_map:
                    prev_idx = self.router_map[router_key].server_idx
                    assert prev_idx == i, f"{router_key}: {prev_idx} vs {i}"
                else:
                    self.router_map[router_key] = sub_server

    def _response(self, request):
        """
        _response
        """
        server = self.router_map[request.router_key]
        response = server.response(request)
        if response.exception is not None:
            raise response.exception
        else:
            return response.value

    def call(self, router_key, name, args=tuple(), kwargs=dict()):
        """
        IPC call method
        """
        request = MethodRequest(
            router_key=router_key, name=name, args=args, kwargs=kwargs
        )
        return self._response(request)

    def attr(self, router_key, name):
        """
        IPC get attribute
        """
        request = AttrRequest(router_key=router_key, name=name)
        return self._response(request)

    def close(self):
        """
        IPC close server
        """
        for server in self.sub_servers:
            if server is not None:
                server.close()

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
The AADiff decorator.
"""
import os
import paddle
import decorator


def get_md5(tensors):
    """
    Get MD5 of tensor, list of tensors or the combination of them.
    """
    if tensors is None:
        return None
    elif isinstance(tensors, paddle.Tensor):
        return tensors._md5sum()
    elif isinstance(tensors, (list, tuple)):
        return [get_md5(t) for t in tensors]
    else:
        raise ValueError(tensors)


def check_aadiff(ntimes=None):
    """
    The AADiff decorator.
    """
    if ntimes is None:
        ntimes = int(os.getenv("AADIFF_TIMES", "0"))

    @decorator.decorator
    def __impl__(_func, *args, **kwargs):
        if ntimes > 0:
            with paddle.no_grad():
                old_md5 = None
                for idx in range(ntimes):
                    ret = _func(*args, **kwargs)
                    print("AADiff Pass {}/{} ...".format(idx, ntimes))
                    cur_md5 = get_md5(ret)
                    del ret
                    if old_md5 is None:
                        old_md5 = cur_md5
                    else:
                        assert old_md5 == cur_md5, "Rank {} has aadiff".format(
                            paddle.distributed.get_rank()
                        )

        return _func(*args, **kwargs)

    return __impl__

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

def _fetch_from_remote(url, force_download=False):
    import hashlib, tempfile, requests, tarfile
    sig = hashlib.md5(url.encode('utf8')).hexdigest()
    cached_dir = os.path.join(tempfile.gettempdir(), sig)
    if force_download or not os.path.exists(cached_dir):
        with tempfile.NamedTemporaryFile() as f:
            #url = 'https://ernie.bj.bcebos.com/ERNIE_stable.tgz'
            r = requests.get(url, stream=True)
            total_len = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_len // 1024, desc='downloading %s' % url, unit='KB'):
                if chunk:
                    f.write(chunk)  
                    f.flush()
            log.debug('extacting... to %s' % f.name)
            with tarfile.open(f.name)  as tf:
                tf.extractall(path=cached_dir)
    log.debug('%s cached in %s' % (url, cached_dir))
    return cached_dir


def add_docstring(doc):
    def func(f):
        f.__doc__ += ('\n======other docs from supper class ======\n%s' % doc)
        return f
    return func


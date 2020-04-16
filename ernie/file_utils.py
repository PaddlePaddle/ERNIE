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
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_len // 1024, desc='downloading %s' % url):
                if chunk:
                    f.write(chunk)  
                    f.flush()
            log.debug('extacting... to %s' % f.name)
            with tarfile.open(f.name)  as tf:
                tf.extractall(path=cached_dir)
    log.debug('%s cached in %s' % (url, cached_dir))
    return cached_dir


"""
Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import copy
import pickle
import lmdb # install lmdb by "pip install lmdb"
import base64

class ImageFeaturesH5Reader(object):
    """
    Reader class
    """
    def __init__(self, features_path):
        self.features_path = features_path
        self.env = lmdb.open(self.features_path, max_readers=1, readonly=True,
                            lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn: 
            self._image_ids = pickle.loads(txn.get('keys'.encode()))

        self.features = [None] * len(self._image_ids)
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()
        index = self._image_ids.index(image_id)
        # Read chunk from file everytime if not loaded in memory.    
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            image_id = item['image_id']
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            num_boxes = int(item['num_boxes'])

            features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)
            g_feat = np.sum(features, axis=0) / num_boxes
            num_boxes = num_boxes + 1
            features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)
            image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            image_location[:, :4] = boxes
            image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) *   \
                    (image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))

            image_location_ori = copy.deepcopy(image_location)
            image_location[:, 0] = image_location[:, 0] / float(image_w)
            image_location[:, 1] = image_location[:, 1] / float(image_h)
            image_location[:, 2] = image_location[:, 2] / float(image_w)
            image_location[:, 3] = image_location[:, 3] / float(image_h)

            g_location = np.array([0, 0, 1, 1, 1])
            image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

            g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
            image_location_ori = np.concatenate([np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0)

        data_json = {"features": features,
                     "num_boxes": num_boxes,
                     "image_location": image_location,
                     "image_location_ori": image_location_ori
            }
        return data_json


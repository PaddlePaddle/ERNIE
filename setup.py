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

import setuptools

from io import open

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddle-ernie",  # Replace with your own username
    version="0.1.0dev1",
    author="Baidu Ernie Team",
    author_email="ernieernie.team@gmail.com",
    description="A pretrained NLP model for every NLP tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/ERNIE/",
    packages=['ernie'],
    install_requires=[
        'requests',
        'tqdm',
        'pathlib2',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], )

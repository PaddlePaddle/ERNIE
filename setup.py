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

"""_summary_

Returns:
    _type_: _description_
"""
import os
import re

from setuptools import find_packages, setup


def get_version() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    with open(os.path.join("erniekit", "utils", "env.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_console_scripts() -> list[str]:
    """_summary_

    Returns:
        list[str]: _description_
    """
    console_scripts = ["erniekit = erniekit.cli:main"]

    return console_scripts


def main():
    """_summary_"""
    setup(
        name="erniekit",
        version=get_version(),
        license="Apache 2.0 License",
        packages=find_packages(include=["erniekit*"]),
        include_package_data=True,
        entry_points={"console_scripts": get_console_scripts()},
    )


if __name__ == "__main__":
    main()

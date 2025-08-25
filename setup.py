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
import io

from setuptools import find_packages, setup


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


REQUIRED_PACKAGES = read_requirements_file("requirements/gpu/requirements.txt")


def get_version() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    with open(os.path.join("erniekit", "version", "env.py"), encoding="utf-8") as f:
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
        maintainer="PaddlePaddle",
        maintainer_email="Paddle-better@baidu.com",
        description="The official repository for ERNIE 4.5 and ERNIEKit - its industrial-grade development toolkit based on PaddlePaddle.",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/PaddlePaddle/ERNIE",
        license="Apache 2.0 License",
        license_files=("LICENSE",),
        packages=find_packages(
            where=".",
            exclude=("tests*", "examples*", "cookbook*"),
        ),
        include_package_data=True,
        install_requires=REQUIRED_PACKAGES,
        entry_points={"console_scripts": get_console_scripts()},
    )


if __name__ == "__main__":
    main()

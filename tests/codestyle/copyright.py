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
"""Copyright file"""

import argparse
import datetime
import os
import re
import sys

COPYRIGHT = """Copyright (c) {year} PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


def _generate_copyright(comment_mark):
    """
    生成版权信息。

    Args:
        comment_mark (str): 注释标记，如'#'或'//'。

    Returns:
        List[str]: 包含版权信息的字符串列表，每个元素是一行注释。
    """
    year = datetime.datetime.now().year
    copyright = COPYRIGHT.format(year=year)

    return [
        (
            f"{comment_mark} {line}{os.linesep}"
            if line
            else f"{comment_mark}{os.linesep}"
        )
        for line in copyright.splitlines()
    ]


def _get_comment_mark(path):
    """
    获取文件路径对应的注释符号，支持 Python、C/C++、Go、Protobuf 等语言类型。

    Args:
        path (str): 文件路径字符串。

    Returns:
        Union[str, None]: 返回一个字符串，表示该文件类型对应的注释符号；如果不支持该文件类型，则返回 None。
    """
    lang_type = re.compile(r"\.(py|pyi|sh)$")
    if lang_type.search(path) is not None:
        return "#"

    lang_type = re.compile(r"\.(h|c|hpp|cc|cpp|cu|go|cuh|proto)$")
    if lang_type.search(path) is not None:
        return "//"

    return None


RE_ENCODE = re.compile(r"^[ \t\v]*#.*?coding[:=]", re.IGNORECASE)
RE_COPYRIGHT = re.compile(r".*Copyright \(c\) \d{4}", re.IGNORECASE)
RE_SHEBANG = re.compile(r"^[ \t\v]*#[ \t]?\!")


def _check_copyright(path):
    """
    检查文件是否包含版权信息，返回布尔值。

    Args:
        path (str): 需要检查的文件路径。

    Returns:
        bool: 如果文件包含版权信息，则返回True；否则返回False。
    """
    head = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = [next(f) for x in range(4)]
    except StopIteration:
        pass

    for idx, line in enumerate(head):
        if RE_COPYRIGHT.search(line) is not None:
            return True

    return False


def generate_copyright(path, comment_mark):
    """
    在文件开头插入版权信息。如果文件已经有版权信息，则不会覆盖原有的版权信息。

    Args:
        path (str): 需要插入版权信息的文件路径。
        comment_mark (str, optional): 注释标记符，默认为'#'。

    Returns:
        None. 直接修改文件内容。
    """
    original_contents = open(path, "r", encoding="utf-8").readlines()
    head = original_contents[0:4]

    insert_line_no = 0
    for i, line in enumerate(head):
        if RE_ENCODE.search(line) or RE_SHEBANG.search(line):
            insert_line_no = i + 1

    copyright = _generate_copyright(comment_mark)
    if insert_line_no == 0:
        new_contents = copyright
        if len(original_contents) > 0 and len(original_contents[0].strip()) != 0:
            new_contents.append(os.linesep)
        new_contents.extend(original_contents)
    else:
        new_contents = original_contents[0:insert_line_no]
        new_contents.append(os.linesep)
        new_contents.extend(copyright)
        if (
            len(original_contents) > insert_line_no
            and len(original_contents[insert_line_no].strip()) != 0
        ):
            new_contents.append(os.linesep)
        new_contents.extend(original_contents[insert_line_no:])
    new_contents = "".join(new_contents)

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(new_contents)


def main(argv=None):
    """
    主函数，用于检查和生成copyright声明。

    Args:
        argv (list, optional): 命令行参数列表；默认为None，使用sys.argv。

    Returns:
        int: 返回值为0表示所有文件都通过了检查，否则返回1。

    Raises:
        无。
    """
    parser = argparse.ArgumentParser(description="Checker for copyright declaration.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    for path in args.filenames:
        comment_mark = _get_comment_mark(path)
        if comment_mark is None:
            print("warning:Unsupported file", path, file=sys.stderr)
            continue

        if _check_copyright(path):
            continue

        generate_copyright(path, comment_mark)


if __name__ == "__main__":
    sys.exit(main())

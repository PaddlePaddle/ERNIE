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
The multilingual text content of prompts, warnings and error messages in the event and
"""

from manager import manager

ALERTS = {
    "chat_check_load_model": {
        "tips": {
            "zh": "✅ 模型已加载成功, server地址为",
            "en": "✅ model loaded successfully， server address is",
        },
        "error": {
            "zh": "❌ 检测模型未加载成功，请检查端口号 {} 是否正常运行",
            "en": "❌The model failed to load. Please check if the port number {} is running normally.",
        },
    },
    "limit_preview_dataset_page": {
        "warning": {
            "zh": "预览数据集页数限制在100页以内，100页后内容不展示",
            "en": "preview data page limit in 100 pages",
        }
    },
    "chat_role_setting": {
        "warning": {
            "zh": "思考模型下角色设置无效",
            "en": "thinking model role setting invalid",
        }
    },
    "chat_system_prompt": {
        "warning": {
            "zh": "思考模型下系统提示词无效",
            "en": "thinking model system prompt invalid",
        }
    },
    "export_split_start": {
        "info": {
            "zh": "开始执行erniekit sever， 请稍候",
            "en": "start run erniekit sever export",
        }
    },
    "export_split_non_existent": {
        "error": {
            "zh": "执行erniekit sever失败, 输入文件夹本地不存在，请检查文件配置",
            "en": "run erniekit sever failed，input folder does not exist,please check file config",
        }
    },
    "export_split_success": {
        "info": {
            "zh": "erniekit sever执行完成",
            "en": "erniekit sever export completed",
        }
    },
    "export_split_find_exceed_file": {
        "info": {
            "zh": "文件 '{}' 大小为 {:.2f}GB，超过 {}GB 限制，开始分割...",
            "en": "The file '{}' has a size of {:.2f} GB, "
            "which exceeds the {} GB limit. Starting the splitting process...",
        }
    },
    "export_split_fail": {"error": {"zh": "erniekit sever执行失败", "en": "erniekit sever export failed"}},
    "preview_data_non_json": {
        "error": {
            "zh": " 文件 {} 不是有效的JSON/JSONL格式，已跳过",
            "en": " File {} is not valid JSON or JSONL format and skipped",
        }
    },
    "preview_data_non_prob": {
        "warning": {
            "zh": " 数据集 {} 的概率prob格式存在问题，已跳过该数据集，请检查dataset_info.json",
            "en": "The probability prob format of the dataset {} is incorrect,"
            " so this dataset has been skipped. Please check dataset_info.json",
        }
    },
    "preview_data_non_type": {
        "warning": {
            "zh": " 数据集 {} 的概率type格式存在问题，已跳过该数据集，请检查dataset_info.json",
            "en": "The probability type format of the dataset {} is incorrect,"
            " so this dataset has been skipped. Please check dataset_info.json",
        }
    },
    "preview_data_non_path": {
        "warning": {
            "zh": " 数据集 {} 的地址path格式存在问题，已跳过该数据集，请检查dataset_info.json",
            "en": "The address path format of the dataset {} is incorrect,"
            " so this dataset has been skipped. Please check dataset_info.json",
        }
    },
    "preview_data_non_existent": {
        "error": {
            "zh": "{} 指向的文件不存在，请检查路径是否正确",
            "en": "The file pointed to by {} does not exist. Please check if the path is correct.",
        }
    },
    "preview_data_error": {
        "error": {
            "zh": " 读取文件 {} 时发生未知错误: {}，已跳过",
            "en": " An unknown error occurred while reading file {}: {}, skipped",
        }
    },
    "preview_data_path_none": {
        "warning": {
            "zh": "请输入数据集或数据地址后再点击预览",
            "en": "Please enter the dataset or data address before clicking preview",
        }
    },
    "progress": {
        "run_command": {"zh": "{} 执行命令: {}", "en": "{} execute command: {}"},
        "user_terminated": {
            "zh": "❌ 进程已被用户终止",
            "en": "❌ The process has been terminated by the user",
        },
        "force_terminated": {
            "zh": "进程未响应，已强制终止",
            "en": "The process is not responding and has been forcibly terminated",
        },
        "terminate_error": {
            "zh": "⚠️ 终止进程时出错: {}",
            "en": "⚠️ An error occurred while terminating the process",
        },
        "progress_end": {
            "zh": "ℹ️ 进程已经结束",
            "en": "ℹ️ The process has already ended",
        },
        "no_progress": {
            "zh": "ℹ️ 没有正在运行的进程",
            "en": "ℹ️ There is no running process",
        },
        "progress_success": {
            "zh": "✅ 命令成功执行完毕",
            "en": "✅ The command has been successfully executed",
        },
        "execution_error": {"zh": "⚠️ 执行错误", "en": "⚠️ Execution error"},
    },
    "max_steps_notice": {
        "info": {
            "zh": "如果最大训练步数大于0，训练轮数设置不起作用",
            "en": "If max steps is greater than 0, the setting of training epochs will not take effect.",
        }
    },
    "thought_model_notice": {"info": {"zh": "该模型不建议微调", "en": "This model should not be fine-tuned"}},
    "custom_model_notice": {
        "info": {"zh": "自定义模型，请自主填入模型路径", "en": "Custom models, please fill in your own model paths"}
    },
    "compute_type_fp8_notice": {
        "warning": {
            "zh": "请注意, fp8仅支持H卡类型的GPU环境",
            "en": "Please note that fp8 only supports GPU environments of H-card type",
        }
    },
}


class Alerts:

    def __init__(self, language="zh"):
        self.language = language
        self.manager = None

    def get(self, key, type, lang=None):
        """
        Retrieve localized alert message from predefined mappings.

        Args:
            self: Instance reference
            key (str): Primary identifier for the alert group
            type (str): Secondary identifier for the alert type within the group
            lang (str, optional): Language code to override default (default: None)

        Returns:
            str: Localized alert message if found, empty string otherwise
        """

        try:
            language = manager.get_component_value("basic", "language")
            return ALERTS[key][type][language if lang is None else lang]
        except Exception as e:
            print(f"Error fetching alert: {e}")
            return ""


alert = Alerts()

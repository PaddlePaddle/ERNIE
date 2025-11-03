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
Multilingual switching for component names and descriptions
"""

chat_info_html = """
        <div style="
            background-color: #e3f2fd; /* 淡蓝色背景 */
            border-radius: 8px;
            padding: 3px 3px; /* 内边距调整 */
            margin: 0px 0; /* 上下边距，左右与页面平齐 */
            text-align: center; /* 文本居中 */
            font-weight: 600; /* 文字加粗 */
            font-size: 16px; /* 增大字体 */
            backdrop-filter: blur(2px); /* 背景虚化效果 */
            -webkit-backdrop-filter: blur(2px); /* 兼容Safari */
        ">
            <div style="
                max-width: 1000px; /* 内容最大宽度 */
                margin: 0 auto; /* 内容居中 */
            ">
                <p style="color: #37474f; margin: 8px 0 0;"> {}</p>
            </div>
        </div>
"""

LOCALES = {
    "chat_tab": {
        "zh": {
            "label": "对话",
        },
        "en": {
            "label": "Chat",
        },
    },
    "preview_dataset": {
        "zh": {"label": "预览数据集", "info": "预览数据集"},
        "en": {"label": "Preview dataset", "info": "Preview dataset"},
    },
    "language": {
        "zh": {"label": "语言", "info": "当前界面支持的语言类型"},
        "en": {"label": "Language", "info": "Settings Language"},
    },
    "model_name": {
        "zh": {"label": "模型名称", "info": "模型名称，Customization为自定义模型"},
        "en": {
            "label": "model name",
            "info": "Model name, Customization means a custom model",
        },
    },
    "model_name_or_path": {
        "zh": {"label": "模型路径", "info": "模型的具体路径"},
        "en": {"label": "model path", "info": "The specific path of the model"},
    },
    "fine_tuning": {
        "zh": {"label": "微调方式", "info": "选择全参、LoRA等训练方式"},
        "en": {"label": "fine tuning", "info": "Settings related to fine tuning"},
    },
    "compute_type": {
        "zh": {"label": "计算类型", "info": "选择计算类型"},
        "en": {"label": "compute type", "info": "Settings related to compute type"},
    },
    "tensor_parallel_degree": {
        "zh": {"label": "tensor parallel degree", "info": "张量并行度"},
        "en": {
            "label": "tensor parallel degree",
            "info": "Settings related to tensor parallel degree",
        },
    },
    "pipeline_parallel_degree": {
        "zh": {
            "label": "pipeline parallel degree",
            "info": "流水线并行度",
        },
        "en": {
            "label": "pipeline parallel degree",
            "info": "Settings related to pipeline parallel degree",
        },
    },
    "sharding_parallel_degree": {
        "zh": {"label": "sharding parallel degree", "info": "分组切片并行度"},
        "en": {
            "label": "sharding parallel degree",
            "info": "Settings related to sharding parallel degree",
        },
    },
    "use_sp_callback": {
        "zh": {"label": "use sp callback", "info": "是否使用sp callback"},
        "en": {
            "label": "use sp callback",
            "info": "whether to use sp callback",
        },
    },
    "port": {
        "zh": {"label": "模型部署端口号", "info": "模型server端口号"},
        "en": {"label": "port", "info": "Settings related to port"},
    },
    "max_model_len": {
        "zh": {"label": "最大生成长度", "info": "对话最大生成的长度"},
        "en": {"label": "max model len", "info": "Settings related to max model len"},
    },
    "top_p": {
        "zh": {"label": "Top-p", "info": "Top-p的采样率取值"},
        "en": {"label": "top-p", "info": "Settings related to top-p"},
    },
    "temperature": {
        "zh": {"label": "温度系数", "info": "对话的温度数"},
        "en": {"label": "temperature", "info": "Settings related to temperature"},
    },
    "eval_dataset_path": {
        "zh": {
            "label": "评测数据路径",
            "info": "关于评测数据路径的相关设置",
            "placeholder": "多个路径请以 , 隔开",
        },
        "en": {
            "label": "eval dataset path",
            "info": "Settings related to eval dataset path",
            "placeholder": "please separate them with ,",
        },
    },
    "eval_dataset_prob": {
        "zh": {
            "label": "评测数据概率",
            "info": "关于评测数据概率的相关设置",
            "placeholder": "多个概率请以 , 隔开",
        },
        "en": {
            "label": "eval dataset prob",
            "info": "Settings related to eval dataset prob",
            "placeholder": "please separate them with ,",
        },
    },
    "max_seq_len": {
        "zh": {"label": "截断长度", "info": "输入序列分词后的最大长度"},
        "en": {"label": "max seq len", "info": "Settings related to max seq len"},
    },
    "batch_size": {
        "zh": {"label": "批处理大小", "info": "每个 GPU 处理的样本数量"},
        "en": {"label": "batch size", "info": "Settings related to batch size"},
    },
    "max_shard_size": {
        "zh": {"label": "最大分块大小(GB)", "info": "单个模型文件的最大大小"},
        "en": {
            "label": "max shard size(GB)",
            "info": "Settings related to max shard size",
        },
    },
    "output_dir_view": {
        "zh": {"label": "导出目录", "info": "保存导出模型的文件夹路径"},
        "en": {"label": "output dir", "info": "Settings related to output dir"},
    },
    "hf_hub_id": {
        "zh": {
            "label": "HF Hub ID（选填）",
            "info": "用于将模型上传至 Hugging Face Hub 的仓库 ID",
        },
        "en": {"label": "hf hub id", "info": "Settings related to hf hub id"},
    },
    "train_dataset": {
        "zh": {"label": "训练数据集选择", "info": "选择内置的训练数据集"},
        "en": {"label": "train dataset", "info": "Settings related to train dataset"},
    },
    "eval_dataset": {
        "zh": {"label": "评估数据集选择", "info": "选择内置的评估数据集"},
        "en": {"label": "eval dataset", "info": "Settings related to eval dataset"},
    },
    "train_dataset_path": {
        "zh": {
            "label": "自定义训练数据路径",
            "info": "关于自定义训练数据路径的相关设置",
            "placeholder": "多个路径请以 , 隔开",
        },
        "en": {
            "label": "customize train dataset path",
            "info": "Settings related to customize train dataset path",
            "placeholder": "please separate them with ,",
        },
    },
    "train_dataset_prob": {
        "zh": {
            "label": "训练数据概率",
            "info": "关于训练数据概率的相关设置",
            "placeholder": "多个概率请以 , 隔开",
        },
        "en": {
            "label": "train dataset prob",
            "info": "Settings related to train dataset prob",
            "placeholder": "please separate them with ,",
        },
    },
    "num_samples_each_epoch": {
        "zh": {
            "label": "虚拟的训练轮数",
            "info": "建议保持默认值",
        },
        "en": {
            "label": "Virtual number of training epochs",
            "info": "It is recommended to maintain the default value",
        },
    },
    "num_train_epochs": {
        "zh": {"label": "训练轮数", "info": "最大训练步数大于零时失效"},
        "en": {
            "label": "num train epochs",
            "info": "Failure occurs when the maximum number of training steps is greater than zero",
        },
    },
    "gradient_accumulation_steps": {
        "zh": {"label": "梯度累积", "info": "梯度累积的步数"},
        "en": {
            "label": "gradient accumulation steps",
            "info": "Settings related to gradient accumulation steps",
        },
    },
    "max_steps": {
        "zh": {"label": "最大训练步数", "info": "设置成-1会自动计算"},
        "en": {"label": "max steps", "info": "Settings related to max steps"},
    },
    "recompute": {
        "zh": {"label": "重计算", "info": "是否使用重计算"},
        "en": {"label": "recompute", "info": "Settings related to recompute"},
    },
    "lora_rank": {
        "zh": {"label": "lora rank", "info": "lora秩设置"},
        "en": {"label": "lora rank", "info": "Settings related to lora rank"},
    },
    "lora_alpha": {
        "zh": {"label": "lora alpha", "info": "lora alpha设置"},
        "en": {"label": "lora alpha", "info": "Settings related to lora alpha"},
    },
    "lora_plus_scale": {
        "zh": {"label": "lora plus scale", "info": "Lora B scale in LoRA+"},
        "en": {"label": "lora plus scale", "info": "Lora B scale in LoRA+"},
    },
    "rslora": {
        "zh": {"label": "rslora", "info": "是否使用 RsLoRA"},
        "en": {"label": "rslora", "info": "Whether to use RsLoRA"},
    },
    "use_quick_lora": {
        "zh": {"label": "use quick lora", "info": "是否使用quick lora"},
        "en": {"label": "use quick lora", "info": "Whether to use quick lora"},
    },
    "dataloader_num_workers": {
        "zh": {"label": "加载数据进程数", "info": "关于加载数据进程数的相关设置"},
        "en": {
            "label": "dataloader num workers",
            "info": "Settings related to dataloader num workers",
        },
    },
    "distributed_dataloader": {
        "zh": {
            "label": "是否启用分布式的dataloader",
            "info": "关于是否启用分布式的dataloader的相关设置",
        },
        "en": {
            "label": "distributed dataloader",
            "info": "Settings related to distributed dataloader",
        },
    },
    "learning_rate": {
        "zh": {"label": "学习率", "info": "AdamW 优化器的初始学习率"},
        "en": {"label": "learning rate", "info": "Settings related to learning rate"},
    },
    "lr_scheduler_type": {
        "zh": {"label": "学习率类型", "info": "关于学习率类型的相关设置"},
        "en": {
            "label": "lr scheduler type",
            "info": "Settings related to lr scheduler type",
        },
    },
    "min_lr": {
        "zh": {"label": "最小的学习率", "info": "关于最小的学习率的相关设置"},
        "en": {"label": "min lr", "info": "Settings related to min lr"},
    },
    "layerwise_lr_decay_bound": {
        "zh": {
            "label": "分层学习率衰减系数",
            "info": "设置成1.0表示不开启",
        },
        "en": {
            "label": "layerwise lr decay bound",
            "info": "Setting it to 1.0 indicates not enabled",
        },
    },
    "weight_decay": {
        "zh": {
            "label": "AdamW优化器权重衰减系数",
            "info": "关于AdamW优化器权重衰减系数的相关设置",
        },
        "en": {"label": "weight decay", "info": "Settings related to weight decay"},
    },
    "adam_epsilon": {
        "zh": {
            "label": "AdamW优化器Epsilon系数",
            "info": "关于AdamW优化器Epsilon系数的相关设置",
        },
        "en": {"label": "adam epsilon", "info": "Settings related to adam epsilon"},
    },
    "adam_beta1": {
        "zh": {
            "label": "AdamW优化器Beta1系数",
            "info": "关于AdamW优化器Beta1系数的相关设置",
        },
        "en": {"label": "adam beta1", "info": "Settings related to adam beta1"},
    },
    "adam_beta2": {
        "zh": {
            "label": "AdamW优化器Beta2系数",
            "info": "关于AdamW优化器Beta2系数的相关设置",
        },
        "en": {"label": "adam beta2", "info": "Settings related to adam beta2"},
    },
    "warmup_steps": {
        "zh": {"label": "预热步数", "info": "学习率预热采用的步数"},
        "en": {"label": "warmup steps", "info": "Number of steps used for warmup."},
    },
    "save_steps": {
        "zh": {"label": "保存间隔", "info": "每两次断点保存间的更新步数"},
        "en": {
            "label": "save steps",
            "info": "Number of steps between two checkpoints.",
        },
    },
    "logging_steps": {
        "zh": {"label": "日志间隔", "info": "每两次日志输出间的更新步数"},
        "en": {"label": "logging steps", "info": "Number of steps between two logs."},
    },
    "save_strategy": {
        "zh": {
            "label": "checkpoint保存策略",
            "info": "关于checkpoint保存策略的相关设置",
        },
        "en": {
            "label": "checkpoint save strategy",
            "info": "Settings related to checkpoint save strategy",
        },
    },
    "evaluation_strategy": {
        "zh": {"label": "评估策略", "info": "关于评估策略的相关设置"},
        "en": {
            "label": "evaluation strategy",
            "info": "Settings related to evaluation strategy",
        },
    },
    "eval_steps": {
        "zh": {"label": "评估间隔", "info": "每两次评估间的更新步数"},
        "en": {"label": "eval steps", "info": "Settings related to eval steps"},
    },
    "out_dir": {
        "zh": {"label": "输出目录", "info": "保存结果的路径"},
        "en": {"label": "out dir", "info": "Directory for saving checkpoints."},
    },
    "save_total_limit": {
        "zh": {
            "label": "最多保存的Checkpoint数量",
            "info": "关于最多保存的Checkpoint数量的相关设置",
        },
        "en": {
            "label": "save total limit",
            "info": "Settings related to save total limit",
        },
    },
    "amp_master_grad": {
        "zh": {
            "label": "amp master grad",
            "info": "amp opt level为O2，使用float32权重梯度",
        },
        "en": {
            "label": "amp master grad",
            "info": "For amp opt level=’O2’, whether to use float32 weight gradients",
        },
    },
    "pipeline_parallel_config": {
        "zh": {
            "label": "pipeline parallel config",
            "info": "pipeline parallel的相关配置",
        },
        "en": {
            "label": "pipeline parallel config",
            "info": "Pipeline parallel related configuration",
        },
    },
    "pp_seg_method": {
        "zh": {
            "label": "pp seg method",
            "info": "流水线层的切分方法",
        },
        "en": {
            "label": "pp seg method",
            "info": "The method used to segment the pipeline layers among pipeline stages",
        },
    },
    "sharding": {
        "zh": {"label": "sharding", "info": "参数分片阶段"},
        "en": {
            "label": "sharding",
            "info": "Parameter sharding stage",
        },
    },
    "moe_group": {
        "zh": {"label": "moe group", "info": "moe 的通信组"},
        "en": {"label": "moe group", "info": "The communication group of moe"},
    },
    "disable_ckpt_quant": {
        "zh": {"label": "disable ckpt quant", "info": "是否禁用checkpoint量化"},
        "en": {
            "label": "disable ckpt quant",
            "info": "Whether disable checkpoint quantization.",
        },
    },
    "max_prompt_len": {
        "zh": {"label": "最大prompt长度", "info": "最大的prompt长度"},
        "en": {"label": "max prompt len", "info": "Maximum prompt length."},
    },
    "release_grads": {
        "zh": {"label": "release grads", "info": "训练时是否释放梯度"},
        "en": {
            "label": "release grads",
            "info": "Whether to release gradients during training",
        },
    },
    "offload_optim": {
        "zh": {"label": "offload optim", "info": "在optimizer.step之后offload优化器"},
        "en": {
            "label": "offload optim",
            "info": "Offload optimizer after optimizer.step()",
        },
    },
    "optim": {
        "zh": {"label": "优化器", "info": "optimizer"},
        "en": {"label": "optimizer", "info": "The optimizer to use"},
    },
    "scale_loss": {
        "zh": {"label": "scale loss", "info": "fp16的初始scale_loss值"},
        "en": {
            "label": "scale loss",
            "info": "The value of initial scale_loss for fp16.",
        },
    },
    "train_tab": {
        "zh": {
            "label": "训练",
        },
        "en": {
            "label": "Train",
        },
    },
    "distributed_parameters_tab": {
        "zh": {
            "label": "分布式参数",
        },
        "en": {
            "label": "Distributed parameters",
        },
    },
    "dataloader_parameters_tab": {
        "zh": {
            "label": "dataloader参数",
        },
        "en": {
            "label": "Dataloader parameters",
        },
    },
    "optimizer_parameters_tab": {
        "zh": {
            "label": "优化器参数",
        },
        "en": {
            "label": "Optimizer parameters",
        },
    },
    "other_parameters_tab": {
        "zh": {
            "label": "其他参数",
        },
        "en": {
            "label": "Other parameters",
        },
    },
    "model_output_tab": {
        "zh": {
            "label": "模型输出",
        },
        "en": {
            "label": "Model output",
        },
    },
    "export_tab": {
        "zh": {
            "label": "导出",
        },
        "en": {
            "label": "Export",
        },
    },
    "eval_tab": {
        "zh": {
            "label": "评估",
        },
        "en": {
            "label": "Eval",
        },
    },
    "best_config": {
        "zh": {"label": "切换SFT/DPO推荐配置", "info": "选择后将覆盖训练配置"},
        "en": {
            "label": "Switch to recommended configuration",
            "info": "Selecting an option will overwrite the training configuration",
        },
    },
    "train_dataset_preview_btn": {
        "zh": {
            "value": "预览训练数据集",
        },
        "en": {
            "value": "Preview Train Dataset",
        },
    },
    "preview_command_btn": {
        "zh": {
            "value": "预览命令行",
        },
        "en": {
            "value": "Preview Command",
        },
    },
    "start_btn": {
        "zh": {
            "value": "开始",
        },
        "en": {
            "value": "Start",
        },
    },
    "stop_btn": {
        "zh": {
            "value": "停止",
        },
        "en": {
            "value": "Stop",
        },
    },
    "clear_btn": {
        "zh": {
            "value": "清空",
        },
        "en": {
            "value": "Clear",
        },
    },
    "command_preview": {
        "zh": {
            "label": "命令行预览",
        },
        "en": {
            "label": "Command preview",
        },
    },
    "output_text": {
        "zh": {
            "label": "命令输出",
        },
        "en": {
            "label": "Output",
        },
    },
    "status_button": {
        "zh": {
            "value": "验证模型加载情况",
        },
        "en": {
            "value": "Verify model loading status",
        },
    },
    "load_model_btn": {
        "zh": {
            "value": "加载模型",
        },
        "en": {
            "value": "Load model",
        },
    },
    "unload_model_btn": {
        "zh": {
            "value": "卸载模型",
        },
        "en": {
            "value": "Unload model",
        },
    },
    "chatbot": {
        "zh": {
            "label": "聊天历史",
        },
        "en": {
            "label": "Chatbot",
        },
    },
    "chat_input": {
        "zh": {
            "label": "输入",
            "placeholder": "请输入内容 注：shift + enter 可直接发送",
        },
        "en": {
            "label": "input",
            "placeholder": "Please enter the content. Note: Press Shift + Enter to send",
        },
    },
    "role_setting": {
        "zh": {"label": "角色设置", "placeholder": "设置角色信息"},
        "en": {"label": "Role setting", "placeholder": "Set up role"},
    },
    "system_prompt": {
        "zh": {"label": "系统提示词", "placeholder": "设置系统提示词"},
        "en": {"label": "System prompt", "placeholder": "Set system prompt"},
    },
    "submit_btn": {
        "zh": {
            "value": "提交",
        },
        "en": {
            "value": "Submit",
        },
    },
    "max_new_tokens": {
        "zh": {
            "label": "最大新生成长度",
        },
        "en": {
            "label": "Max new tokens",
        },
    },
    "start_merge_btn": {
        "zh": {
            "value": "开始merge lora权重",
        },
        "en": {
            "value": "Start merge lora weights",
        },
    },
    "start_split_btn": {
        "zh": {
            "value": "开始split模型",
        },
        "en": {
            "value": "Start split model",
        },
    },
    "checkpoint_path": {
        "zh": {"label": "检查点 Checkpoint", "info": "需要导出的checkpoint路径"},
        "en": {
            "label": "Checkpoint path",
            "info": "The checkpoint path needed for exporting",
        },
    },
    "gpu_num": {
        "zh": {"label": "可用GPU数量", "info": "当前机器可用的GPU数量"},
        "en": {
            "label": "Available GPU number",
            "info": "The available GPU number of this machine",
        },
    },
    "dataset_preview_title": {
        "zh": {"value": "### 数据集预览"},
        "en": {"value": "### Dataset Preview"},
    },
    "page_info": {
        "zh": {"value": "第 {} 页，共 {} 页"},
        "en": {"value": "Page {}, Total Pages {}"},
    },
    "dataset_info": {
        "zh": {"value": "（**数据集** {}/{}） 当前预览地址: {}"},
        "en": {"value": "(**Dataset** {}/{})  Current preview path: {}"},
    },
    "next_btn": {
        "zh": {"value": "下一页"},
        "en": {"value": "Next Page"},
    },
    "prev_btn": {
        "zh": {"value": "上一页"},
        "en": {"value": "Pre Page"},
    },
    "close_button": {
        "zh": {"value": "关闭"},
        "en": {"value": "Close"},
    },
    "next_dataset_btn": {
        "zh": {"value": "下一个数据集"},
        "en": {"value": "Next Dataset"},
    },
    "prev_dataset_btn": {
        "zh": {"value": "上一个数据集"},
        "en": {"value": "Pre Dataset"},
    },
    "chat_info": {
        "zh": {
            "value": chat_info_html.format(
                "⚠️ 对话功能暂不支持加载lora checkpoint，如果需要加载lora checkpoint，"
                "请自行导出模型，并修改模型名称和路径"
            ),
        },
        "en": {
            "value": chat_info_html.format(
                "⚠️ Chat doesn't support lora checkpoint loading, "
                + "Export lora model & set model name/path manually."
            )
        },
    },
    "model_source": {
        "zh": {"label": "模型库", "info": "选择模型的下载库"},
        "en": {"label": "Choose Path", "info": "Choose different paths"},
    },
    "eval_dataset_btn": {
        "zh": {
            "value": "设置评估数据集",
        },
        "en": {
            "value": "Setting Eval Dataset",
        },
    },
    "train_dataset_btn": {
        "zh": {
            "value": "设置训练数据集",
        },
        "en": {
            "value": "Setting Train Dataset",
        },
    },
    "text_dataset_btn": {
        "zh": {
            "value": "设置文本数据集",
        },
        "en": {
            "value": "Setting Text Dataset",
        },
    },
    "add_dataset_btn": {
        "zh": {
            "value": "添加数据集",
        },
        "en": {
            "value": "Add Dataset",
        },
    },
    "save_dataset_btn": {
        "zh": {
            "value": "保存数据集并退出",
        },
        "en": {
            "value": "Save Dataset And Exit",
        },
    },
    "delete_dataset_btn": {
        "zh": {
            "value": "删除数据集",
        },
        "en": {
            "value": "Delete Dataset",
        },
    },
    "eval_dataset_preview_btn": {
        "zh": {
            "value": "预览评估数据集",
        },
        "en": {
            "value": "Preview Eval Dataset",
        },
    },
    "text_dataset_preview_btn": {
        "zh": {
            "value": "预览文本数据集",
        },
        "en": {
            "value": "Preview Text Dataset",
        },
    },
    "dataset": {"zh": {"label": "数据集"}, "en": {"label": "Datasets"}},
    "dataset_type": {
        "zh": {"label": "数据集类型"},
        "en": {
            "label": "Dataset Types",
        },
    },
    "dataset_path": {
        "zh": {"label": "数据集路径"},
        "en": {
            "label": "Dataset Path",
        },
    },
    "dataset_prob": {
        "zh": {"label": "数据集概率"},
        "en": {
            "label": "Dataset Probability",
        },
    },
    "dataset_tip": {
        "zh": {
            "value": "### Tips: 请选择相应的数据集，其中Customization为自定义数据集，填入数据集地址和数据集概率"
        },
        "en": {
            "value": "### Tips: Please select the appropriate dataset, where Customization means custom datasets"
        },
    },
    "open_close_plot_btn": {
        "zh": {"value": "打开/关闭Loss曲线"},
        "en": {"value": "Open/Close Loss Plot"},
    },
    "file_input": {
        "zh": {"label": "点击上传文件"},
        "en": {"label": "Click to upload files"},
    },
    "thought_checkbox": {
        "zh": {
            "label": "开启多模思考模式",
        },
        "en": {
            "label": "Enable VL Thought Mode",
        },
    },
    "pp_need_data_degree": {
        "zh": {
            "label": "pp_need_data_degree",
        },
        "en": {
            "label": "pp_need_data_degree",
        },
    },
    "virtual_pp_degree": {
        "zh": {
            "label": "virtual_pp_degree",
        },
        "en": {
            "label": "virtual_pp_degree",
        },
    },
    "modality_ratio": {
        "zh": {
            "label": "多模态数据比例",
            "info": "多模和纯文数据配比的配置项",
        },
        "en": {
            "label": "modality ratio",
            "info": "Configuration item for multi-modal and pure text data ratio",
        },
    },
    "generate_log_btn": {
        "zh": {"label": "生成对话日志"},
        "en": {"label": "Generate Chat Log"},
    },
    "download_log_btn": {
        "zh": {"label": "下载对话日志"},
        "en": {"label": "Download Chat Log"},
    },
    "chat_role_setting": {
        "zh": {"label": "角色设置"},
        "en": {"label": "Role Setting"},
    },
    "img_url_input": {
        "zh": {
            "label": "URL图片",
            "placeholder": "多个路径请以 ; 隔开, 比如url1;url2;url3",
        },
        "en": {
            "label": "URL Image",
            "placeholder": "Multiple URLs separated by ;, e.g., url1;url2;url3",
        },
    },
    "video_url_input": {
        "zh": {
            "label": "URL视频",
            "placeholder": "多个路径请以 ; 隔开, 比如url1;url2;url3",
        },
        "en": {
            "label": "URL Video",
            "placeholder": "Multiple URLs separated by ;, e.g., url1;url2;url3",
        },
    },
}


def get(key, lang=None, prop=None):
    """
    Retrieves the Chinese translation based on the given key

    Args:
        key (str): The key used to look up the translation.

    Returns:
        str: The Chinese translation corresponding to the key.
    """
    if prop and lang:
        return LOCALES[key][lang][prop]
    return LOCALES[key]["zh"]["label"]

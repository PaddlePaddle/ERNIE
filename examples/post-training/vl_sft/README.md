# 多模 SFT

有监督微调（Supervised Fine-Tuning）是指在预训练的语言模型上，使用有标签的数据集进行进一步的微调，以适应特定的下游任务。这个过程依赖于高质量的标注数据，调整的是模型的所有参数，适用于任务非常具体，模型需要在这些特定任务上表现出高精度的场景。

LoRA（Low-Rank Adaption）一般是在SFT/DPO的基础上，利用矩阵低秩分解的技术，达到只需微调少量新参数便能完成模型微调的目的。LoRA训练能够降低模型训练所需资源，小数据集通常表现接近甚至优于全参训练。

## 1. 数据格式

我们提供了demo数据，数据位于`./data`目录下，您可以基于demo数据进行快速体验，也可以基于自己的数据进行训练。

* 每条训练数据均为json格式，多条数据以换行符进行隔开

```json
{
    "image_info": [
        {
            "bos_url": "bos://fine-code/code_mm_data/0910/sft/118000093524076198.png",
            "matched_text_index": 0,
            "image_width": 551,
            "image_height": 266,
            "url": null,
            "title": "",
            "matched_sim": null
        }
    ],
    "content": "plsql连接报如图错误，怎么解决\n这个错误提示表示PL/SQL Developer无法找到OCI（Oracle Call Interface）DLL文件。这通常是由于Oracle客户端未正确安装或Oracle客户端的路径未正确配置引起的。以下是一些解决方法：\n\n### 解决方法1：检查Oracle客户端安装\n1. 确认已经安装了Oracle客户端软件。\n2. 检查安装目录中是否存在oci.dll文件，通常在`<oracle_home>\\bin`目录下。\n\n### 解决方法2：配置Oracle客户端路径\n1. 在PL/SQL Developer中配置Oracle Home目录和OCI DLL：\n   1. 打开PL/SQL Developer。\n   2. 进入`Tools` -> `Preferences`。\n   3. 选择`Connection`。\n   4. 设置`Oracle Home`为OCI库所在路径，例如：`C:\\oracle\\product\\...\\dbhome_1`。\n   5. 设置`OCI Library`为具体的dll文件路径，例如：`C:\\oracle\\product\\...\\dbhome_1\\bin\\oci.dll`。\n2. 确保已设置环境变量`ORACLE_HOME`和`PATH`，包含Oracle客户端安装路径。\n   1. 右键`此电脑`（或`计算机`），选择`属性`。\n   2. 选择`高级系统设置`。\n   3. 点击`环境变量`。\n   4. 设置`ORACLE_HOME`指向Oracle客户端的安装目录，例如：`C:\\oracle\\product\\...\\dbhome_1`。\n   5. 将Oracle客户端的`bin`目录添加到`PATH`中，例如：`C:\\oracle\\product\\...\\dbhome_1\\bin`。\n\n### 解决方法3：检查32位或64位\n1. 确认PL/SQL Developer和Oracle客户端的位数匹配。如果PL/SQL Developer是32位的，那么必须使用32位的Oracle客户端，64位类似。\n\n### 解决方法4：重新安装或修复Oracle客户端\n1. 如果上述方法都不起作用，考虑重新安装Oracle客户端，并确保选择所有需要的组件。\n\n### 示例：\n假设Oracle客户端安装位置为`C:\\oracle\\product\\12.2.0\\dbhome_1\\`，那么：\n- Oracle Home: `C:\\oracle\\product\\12.2.0\\dbhome_1`\n- OCI Library: `C:\\oracle\\product\\12.2.0\\dbhome_1\\bin\\oci.dll`\n\n进行上述设置后，重新启动PL/SQL Developer并尝试连接。希望这些步骤能帮你解决问题！",
    "id": "58aa270277da25220b44d44621b4b951",
    "source": null,
    "meta_data": {
        "data_type": "sft",
        "is_text_sent_deduplitation": false,
        "is_bos_path_exists": false,
        "is_matched_sim": false,
        "from_source": "ebv_data_sft_code",
        "creator": "zhangruixi",
        "create_time": "20240910",
        "update_time": "20240910",
        "tags": {
            "tag_1": "代码",
            "tag_2": "代码debug"
        },
        "create_method": "O2",
        "data_labeled": true,
        "provider": "nlp",
        "difficulty": "middle",
        "quality": "middle",
        "data_id": "118000093524076198",
        "source_sft_version": "log"
    },
    "text_info": [
        {
            "text": "plsql连接报如图错误，怎么解决",
            "tag": "mask"
        },
        {
            "text": "这个错误提示表示PL/SQL Developer无法找到OCI（Oracle Call Interface）DLL文件。这通常是由于Oracle客户端未正确安装或Oracle客户端的路径未正确配置引起的。以下是一些解决方法：\n\n### 解决方法1：检查Oracle客户端安装\n1. 确认已经安装了Oracle客户端软件。\n2. 检查安装目录中是否存在oci.dll文件，通常在`<oracle_home>\\bin`目录下。\n\n### 解决方法2：配置Oracle客户端路径\n1. 在PL/SQL Developer中配置Oracle Home目录和OCI DLL：\n   1. 打开PL/SQL Developer。\n   2. 进入`Tools` -> `Preferences`。\n   3. 选择`Connection`。\n   4. 设置`Oracle Home`为OCI库所在路径，例如：`C:\\oracle\\product\\...\\dbhome_1`。\n   5. 设置`OCI Library`为具体的dll文件路径，例如：`C:\\oracle\\product\\...\\dbhome_1\\bin\\oci.dll`。\n2. 确保已设置环境变量`ORACLE_HOME`和`PATH`，包含Oracle客户端安装路径。\n   1. 右键`此电脑`（或`计算机`），选择`属性`。\n   2. 选择`高级系统设置`。\n   3. 点击`环境变量`。\n   4. 设置`ORACLE_HOME`指向Oracle客户端的安装目录，例如：`C:\\oracle\\product\\...\\dbhome_1`。\n   5. 将Oracle客户端的`bin`目录添加到`PATH`中，例如：`C:\\oracle\\product\\...\\dbhome_1\\bin`。\n\n### 解决方法3：检查32位或64位\n1. 确认PL/SQL Developer和Oracle客户端的位数匹配。如果PL/SQL Developer是32位的，那么必须使用32位的Oracle客户端，64位类似。\n\n### 解决方法4：重新安装或修复Oracle客户端\n1. 如果上述方法都不起作用，考虑重新安装Oracle客户端，并确保选择所有需要的组件。\n\n### 示例：\n假设Oracle客户端安装位置为`C:\\oracle\\product\\12.2.0\\dbhome_1\\`，那么：\n- Oracle Home: `C:\\oracle\\product\\12.2.0\\dbhome_1`\n- OCI Library: `C:\\oracle\\product\\12.2.0\\dbhome_1\\bin\\oci.dll`\n\n进行上述设置后，重新启动PL/SQL Developer并尝试连接。希望这些步骤能帮你解决问题！",
            "tag": "no_mask"
        }
    ]
}
```

## 2. SFT训练示例

更多训练配置可以参考 [训练通用配置](../../../docs/training_args.md#1-通用配置) 和 [SFT训练专用配置](../../../docs/training_args.md#21-sft专用配置)

### 3.1 启动训练

**示例一. 8K序列长度，SFT**

以下示例需要在96卡 80G A/H机器上完成训练，请确保在12台互联的机器下执行下述脚本，并选择其中一台机器作为主节点，将其IP地址替换 `${HOST}` ，每台机器以相同的命令启动。

```bash
sh examples/post-training/vl_sft/script/train_gpu.sh
```

### 3.2 查看训练日志

如果您的脚本里指定了 `logging_dir` 参数，我们会保存VisualDL的可视化结果到该目录下，否则相关结果保存于 `output_dir` 指定的路径下

通过如下命令启动VisualDL查看训练日志

```bash
visualdl --logdir ${YOUR_LOG_DIR} --host ${HOST_IP} --port ${PORT}
```

### 3.3 合并LoRA参数

如果您在训练完成后，需要将LoRA参数合并到原始模型中，具体命令如下，请将 `${LORA_MODEL_PATH}` `${BASE_MODEL_PATH}` `${OUTPUI_PATH}` 替换成实际目录：

```bash
python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    examples/post-training/tools/mergekit.py \
    --lora_model_path ${LORA_MODEL_PATH} \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --output_path ${OUTPUI_PATH}
```

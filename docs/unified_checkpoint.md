# Unified Checkpoint User Guide

## 1. Overview

**Unified Checkpoint** is a storage solution designed by PaddlePaddle for large model scenarios. Its core idea is to store model weights, optimizer weights, and other parameters in a unified `safetensors` format, without distinguishing between different distributed strategies during saving. This improves the generality of checkpoint storage for large models.

Benefits of using **Unified Checkpoint** include:

* Direct compatibility with downstream tasks. For example, there is no need to manually merge model parameters before performing dynamic graph inference or model deployment.

* Flexibly supports dynamic scaling, allowing seamless transitions across different distributed strategies and varying numbers of training machines. Checkpoints no longer require special handling for different distributed configurations.

* Lossless checkpoint compression combined with asynchronous saving enables fast storage and significantly reduces model storage costs.

## 2. ERNIEKit Training Coverage

| Training Method      | Supported |
|----------------------|-----------|
| SFT                  | Yes       |
| SFT-LoRA             | Yes       |
| DPO                  | Yes       |
| DPO-LoRA             | Yes       |
| Pre-train             | No        |

## 3. Usage Instructions

**Unified Checkpoint** includes two configuration options. The relevant settings are as follows:

```yaml
"unified_checkpoint": True  # Default setting is True
"unified_checkpoint_config": ""  # Default is empty; specific configurable parameters are detailed below.
```

### 3.1 Asynchronous Saving

During large model training, periodic checkpoint saving is commonly enabled to preserve training progress and minimize computational loss in the event of interruptions.  However, in standard saving scenarios, the time required to save is directly proportional to the model size. If the saving interval is short, training tasks may pause frequently, resulting in wasted GPU resources.

To address this, we implement an **Asynchronous Saving Mechanism**, allowing checkpoint saving to overlap with the training process. This reduces GPU idle time caused by checkpoint interruptions and improves overall training efficiency.

**Important**: Asynchronous saving consumes additional CPU memory. Make sure the system has enough available memory before turning on this feature.

To enable this feature, set:
```yaml
unified_checkpoint_config: "async_save"
```

### 3.2 Checkpoint Compression

During large model training, frequent checkpointing is essential for fast recovery after interruptions.
However, large checkpoints consume significant disk space, limiting how many can be retained.
For example, a single checkpoint of a 72B model in BF16 training can take up nearly **1 TB**.

Checkpoint compression addresses this by applying **Int8 (O1)** and **Int4 (O2)** compression to optimizer parameters, reducing storage usage by up to **78.5%** without affecting training quality.

To enable this feature, set:
```yaml
unified_checkpoint_config: "remove_master_weight"  # Toggle to skip saving the master weight. When enabled, the master weight will not be stored in the checkpoint; instead, it will be reconstructed from the model weight during loading.

ckpt_quant_stage: "O1"  # Switch to enable checkpoint compression. `O0`: disables compression; `O1`: applies Int8 compression; `O2`: applies Int4 compression.

disable_ckpt_quant: False  # Disable checkpoint compression, only available for SFT training.
```

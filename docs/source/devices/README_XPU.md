# Kunlunxin XPU ERNIE-4.5-300B-A47B-Base & ERNIE-4.5-300B-A47B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need Kunlun XPU machine, and the system requirements for this machine are as follows:

| Chip type | Driver version |
 | --- | --- |
| KunlunxinP800 | 5.0.21.21 |

#### Instructions for the Minimum Number of XPU Cards Required for Training
SFT: At least 112 cards (14 nodes x 8 cards) of 96G Kunlunxin P800 cards are required.
LoRA: At least 16 cards (2 nodes x 8 cards) of 96G Kunlunxin P800 cards are required.

#### Environment Description
- **Machine：** KunlunxinP800 96GB 8-card machine x 14
- **Docker image：** registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
- **GCC path：** /usr/bin/gcc (8.4)
- **python version：** 3.10
  **Note: This example uses an 8-card machine: To verify if your machine is a Kunlunxin, simply enter the command in the system environment and see if there is any output:**
```
xpu_smi
#example：$ xpu_smi
Wed Jun 25 19:45:10 2025
+-----------------------------------------------------------------------------+
| XPU-SMI               Driver Version: 5.0.21.21    XPU-RT Version: 5.0.21   |
|-------------------------------+----------------------+----------------------+
| XPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | XPU-Util  Compute M. |
|                               |             L3-Usage |            SR-IOV M. |
|===============================+======================+======================|
|   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |                    0 |
| N/A   37C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  P800 OAM           N/A   | 00000000:05:00.0 N/A |                    0 |
| N/A   41C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  P800 OAM           N/A   | 00000000:63:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  P800 OAM           N/A   | 00000000:65:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  P800 OAM           N/A   | 00000000:83:00.0 N/A |                    0 |
| N/A   40C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  P800 OAM           N/A   | 00000000:85:00.0 N/A |                    0 |
| N/A   40C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  P800 OAM           N/A   | 00000000:A3:00.0 N/A |                    0 |
| N/A   39C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  P800 OAM           N/A   | 00000000:A5:00.0 N/A |                    0 |
| N/A   40C  N/A     87W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  XPU   XI   CI        PID   Type   Process name                  XPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-15 minutes)

1. Pull the Image
```

docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```

2. Start the Container
```
# Recommended: Map your project directory and a dataset directory
# Replace pwd with the actual path on your host machine
docker run -it --privileged=true  --net host --shm-size '256gb' --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 --device=/dev/xpuctrl:/dev/xpuctrl --name paddle-xpu-dev -v $(pwd):/work -w=/work -v xxx ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

3. Install paddlepaddle-xpu
```
# The "PaddlePaddle" deep learning framework provides basic computing capabilities
python -m pip install paddlepaddle-xpu==3.3.0.dev20251016 -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
# Paddle_xpu contains a small number of XPU custom operators, mainly used to support XPU training acceleration
wget https://bj.bcebos.com/v1/klx-paddlelite/paddle_whl/paddle_kl3/daily_output/20251014/paddle_xpu-0.0.1-py3-none-any.whl
python -m pip install paddle_xpu-0.0.1-py3-none-any.whl

Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/paddlepaddle-xpu/
```

4. Install requirements
```
pip install -r requirements/gpu/requirements.txt
python -m pip install -e . # We recommend install in editable mode
```

### (2) Start post-traning：(Adjust NIC names for your setup, this will take a relatively long time)

We provided erniekit to run different configurations, blow is an example:
```
erniekit train examples/configs/xpu/ERNIE-4.5-21B-A3B/sft/run_sft_lora_8k.yaml
```
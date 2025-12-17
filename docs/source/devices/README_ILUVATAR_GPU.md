# Iluvatar GPU ERNIE-4.5-21B-A3B-Base & ERNIE-4.5-21B-A3B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need a Iluvatar GPU machine, and the system requirements for this machine are as follows:

 | Chip type | Driver version |
 | --- | --- |
 | BI150 | 4.3.0 |

#### Environment Description
- **Machine：** BI150 64GB 8-card machine
- **Docker image：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
- **GCC path：** /usr/bin/gcc (9.4)
- **python version：** 3.10

**Note: This example uses an 8-card machine: To verify if your machine is Iluvatar GPU, simply enter the command in the system environment and see if there is any output:**
```bash
ixsmi
#example：$ ixsmi
Timestamp    Thu Jul 10 16:59:37 2025
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.0       Driver Version: 4.3.0       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:16:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 2    Iluvatar BI-V150         | 00000000:1C:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 3    Iluvatar BI-V150         | 00000000:1F:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 4    Iluvatar BI-V150         | 00000000:27:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 5    Iluvatar BI-V150         | 00000000:2A:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    105W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 6    Iluvatar BI-V150         | 00000000:34:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 7    Iluvatar BI-V150         | 00000000:37:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 8    Iluvatar BI-V150         | 00000000:3D:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 9    Iluvatar BI-V150         | 00000000:40:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 10   Iluvatar BI-V150         | 00000000:48:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 11   Iluvatar BI-V150         | 00000000:4B:00.0     | 1500MHz   1600MHz    |
| N/A  33C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 12   Iluvatar BI-V150         | 00000000:54:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 13   Iluvatar BI-V150         | 00000000:57:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    104W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 14   Iluvatar BI-V150         | 00000000:64:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 15   Iluvatar BI-V150         | 00000000:67:00.0     | 1500MHz   1600MHz    |
| N/A  36C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-15 minutes)

1. Pull the Image
```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

2. Install the driver kmd on host
```bash
wget https://ai-rank.bj.bcebos.com/Iluvatar/corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
bash corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
```

3. Start the Container
```bash
docker run -itd --name paddle-ixuca-dev -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    -v /dev:/dev -v /home:/home --privileged --cap-add=ALL --pid=host --network=host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle-ixuca-dev bash
```

4. Install paddlepaddle & paddle-iluvatar-gpu
```bash
ln -sf /usr/local/bin/python3 /usr/local/bin/python

# Install PaddlePaddle CPU package
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cpu/paddlepaddle/paddlepaddle-3.3.0.dev20251023-cp310-cp310-linux_x86_64.whl

# Install PaddlePaddle iluvatar-gpu plugin package
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/ixuca/paddle-iluvatar-gpu/paddle_iluvatar_gpu-3.0.0.dev20251023-cp310-cp310-linux_x86_64.whl

Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```

4. Install ERNIEKit
```bash
git clone https://github.com/PaddlePaddle/ERNIE.git
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e . # We recommend install in editable mode
```

### (2) Start post-traning：(This will take a relatively long time)

SFT fine-tuning

```bash
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export PADDLE_XCCL_BACKEND=iluvatar_gpu

rm -rf erniekit_dist_log/ output/ vdl_log/

# ERNIE-4.5-0.3B sft
erniekit train examples/configs/iluvatar_gpu/ERNIE-4.5-0.3B/sft/run_sft_8k.yaml

# ERNIE-4.5-21B sft
erniekit train examples/configs/iluvatar_gpu/ERNIE-4.5-21B-A3B/sft/run_sft_8k.yaml
```

SFT-LoRA fine-tuning

```bash
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export PADDLE_XCCL_BACKEND=iluvatar_gpu

rm -rf erniekit_dist_log/ output/ vdl_log/

# ERNIE-4.5-0.3B sft-lora
erniekit train examples/configs/iluvatar_gpu/ERNIE-4.5-0.3B/sft/run_sft_lora_8k.yaml

# ERNIE-4.5-21B sft-lora
erniekit train examples/configs/iluvatar_gpu/ERNIE-4.5-21B-A3B/sft/run_sft_lora_8k.yaml
```


# Iluvatar GPU ERNIE-4.5-VL-28B-A3B Training Quick Start


##  🚀 Quick Start🚀

### （0）Before starting, you need a Iluvatar GPU machine, and the system requirements for this machine are as follows:

 | Chip type | Driver version |
 | --- | --- |
 | BI150 | 4.3.0 |

#### Environment Description
- **Machine：** BI150 64GB 8-card machine
- **Docker image：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
- **GCC path：** /usr/bin/gcc (9.4)
- **python version：** 3.10

**Note: This example uses an 8-card machine: To verify if your machine is Iluvatar GPU, simply enter the command in the system environment and see if there is any output:**
```
ixsmi
#example：$ ixsmi
Timestamp    Thu Jul 10 16:59:37 2025
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.0       Driver Version: 4.3.0       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:16:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 2    Iluvatar BI-V150         | 00000000:1C:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 3    Iluvatar BI-V150         | 00000000:1F:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 4    Iluvatar BI-V150         | 00000000:27:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 5    Iluvatar BI-V150         | 00000000:2A:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    105W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 6    Iluvatar BI-V150         | 00000000:34:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 7    Iluvatar BI-V150         | 00000000:37:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 8    Iluvatar BI-V150         | 00000000:3D:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 9    Iluvatar BI-V150         | 00000000:40:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 10   Iluvatar BI-V150         | 00000000:48:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 11   Iluvatar BI-V150         | 00000000:4B:00.0     | 1500MHz   1600MHz    |
| N/A  33C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 12   Iluvatar BI-V150         | 00000000:54:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 13   Iluvatar BI-V150         | 00000000:57:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    104W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 14   Iluvatar BI-V150         | 00000000:64:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 15   Iluvatar BI-V150         | 00000000:67:00.0     | 1500MHz   1600MHz    |
| N/A  36C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (1)  Environment Preparation: (This will take you 5-15 minutes)

1. Pull the Image
```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

2. Install the driver kmd on host
```bash
wget https://ai-rank.bj.bcebos.com/Iluvatar/corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
bash corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
```

3. Start the Container
```bash
docker run -itd --name paddle-ixuca-dev -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    -v /dev:/dev -v /home:/home --privileged --cap-add=ALL --pid=host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle-ixuca-dev bash
```

4. Install paddlepaddle & paddle-iluvatar-gpu & FastDeploy
```bash
# Install PaddlePaddle CPU package
pip3 install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddlePaddle iluvatar-gpu plugin package
pip3 install --pre paddle-iluvatar-gpu==3.0.0.dev20250926 -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/

# Install FastDeploy package
pip3 install fastdeploy_iluvatar_gpu==2.3.0.dev0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
```
Nightly version link:
https://www.paddlepaddle.org.cn/packages/nightly/ixuca/

5. Install ERNIEKit
```bash
ln -sf /usr/local/bin/python3 /usr/local/bin/python
git clone https://github.com/PaddlePaddle/ERNIE.git
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e . # We recommend install in editable mode
```

6. Install requirements
```bash
pip3 install allure-pytest
```

7. Download DoclingMatix Datasets
```bash
cd ERNIE
wget https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz
tar zxvf DoclingMatix.tar.gz
mv DoclingMatix examples/data
rm -rf DoclingMatix.tar.gz
```

### (2) Start post-training：(This will take a relatively long time)

SFT-LoRA fine-tuning

```bash
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
export PADDLE_XCCL_BACKEND=iluvatar_gpu

rm -rf erniekit_dist_log/ output/ vdl_log/

erniekit train examples/configs/iluvatar_gpu/ERNIE-4.5-VL-28B-A3B/sft/run_sft_lora_8k.yaml
```

### (3) Run unit test
```bash
bash tests/iluvatar_gpu/run_lora_vl_test.sh
```

# 环境安装与配置

# 1、环境版本要求

- 环境主要安装Python和Paddle对应版本要求的环境，中间建议使用pip安装方式进行安装。
- Python3版本要求：python3.7及以上版本，参考https://www.python.org/ 
- PaddlePaddle版本要求：paddlepaddle2.0+版本，参考https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/macos-pip.html
- Paddle环境的安装，需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构（mac M1除外，paddle 已支持Mac M1 芯片）。下面的第一行输出的是”64bit”，第二行输出的是”x86_64”、”x64”或”AMD64”即可：

```java
python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```

# 2、CPU机器和GPU机器的安装

## 2.1 CPU机器的安装

- 请参考1，完成Paddle和Python3的安装即可

## 2.2 GPU机器的安装

- 使用GPU机器时，比CPU机器的安装多了GPU相关驱动的配置安装

**1、 GPU环境及示例**

- 需要您确认自己的GPU机器的安装情况，包括：**nvidia驱动、cuda版本、cudnn版本、nccl版本。**
- 以下是文心ERNIE开发套件在GPU机器上运行的环境配置示例：

**环境示例**

- Tesla V100上nvidia驱动、cuda版本、cudnn版本、nccl版本、python版本以及PaddlePaddle版本
  - NVIDIA Driver Version: 418.67
    - CUDA Version: 10.1
    - CUDNN Version：7.6.0
    - NCCL Version: 2.3.5
    - Python 3.7.1及以上
    - PaddlePaddle 2.2
- Tesla K40上nvidia驱动、cuda版本、cudnn版本、nccl版本、python版本以及PaddlePaddle版本
  - NVIDIA Driver Version: 418.39
  - CUDA Version: 10.1
  - CUDNN Version：7.0.3
  - NCCL Version: 2.3.5
  - Python 3.7.1及以上
  - PaddlePaddle 2.2

**2、** **配置环境变量：**

- 上述环境配置完成以后，可以参考以下方式进行运行时环境变量的配置。
- 如果您的开发机上已经配置好文心ERNIE开发套件需要的环境，可以参考以下命令设置您的运行环境，配置如下：

```plain
set -x
#在LD_LIBRARY_PATH中添加cuda库的路径
export LD_LIBRARY_PATH=/home/work/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#在LD_LIBRARY_PATH中添加cudnn库的路径
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn7.6.5/lib64:$LD_LIBRARY_PATH
#如果需要多卡并行训练，需要先下载NCCL，下载地址：http://bj.bcebos.com/wenxin-models/nccl.tar.gz，然后在LD_LIBRARY_PATH中添加NCCL库的路径
export LD_LIBRARY_PATH=/home/work/nccl_2.3.5/lib:$LD_LIBRARY_PATH
#如果FLAGS_sync_nccl_allreduce为1，则会在allreduce_op_handle中调用cudaStreamSynchronize（nccl_stream），这种模式在某些情况下可以获得更好的性能
export FLAGS_sync_nccl_allreduce=1
#是否是分布式训练，0标识是分布式，1标识是单机
export PADDLE_IS_LOCAL=1
export PADDLE_USE_GPU=1
#表示分配的显存块占GPU总可用显存大小的比例，范围[0,1]
export FLAGS_fraction_of_gpu_memory_to_use=0.5
#选择要使用的GPU
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
#表示是否使用垃圾回收策略来优化网络的内存使用，<0表示禁用，>=0表示启用
export FLAGS_eager_delete_tensor_gb=1.0
#是否使用快速垃圾回收策略
export FLAGS_fast_eager_deletion_mode=1
#垃圾回收策略释放变量的内存大小百分比，范围为[0.0, 1.0]
export FLAGS_memory_fraction_of_eager_deletion=1
#设置python
#alias python= your python path
#alias pip= your pip path
```

- 注意：如果需要多卡并行训练，需要先下载NCCL，下载地址：http://bj.bcebos.com/wenxin-models/nccl.tar.gz ，然后在LD_LIBRARY_PATH中添加NCCL库的路径

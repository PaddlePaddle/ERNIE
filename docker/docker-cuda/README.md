# Build Docker in NVIDIA GPUs

This directory shows how to build Docker images for ERNIE in NVIDIA GPUs. It contains Docker configuration files for running ERNIE.

Before building Docker images, make sure that you have Docker and NVIDIA Container Toolkit installed on your system.

* Build and run Docker

```bash
# Build the image
docker build -f ./Dockerfile \
    --build-arg HTTP_PROXY=*** \
    --build-arg PIP_INDEX=*** \
    --build-arg BASE_IMAGE=*** \
    -t my_ernie_docker:latest .

# Run the container
docker run --gpus all --name ernie-work \ 
    -v $(pwd):/work  -w=/work --shm-size=512G \
    --network=host -it my_ernie_docker:latest bash

# Enter the container
docker exec -it ernie-work bash
```

You can modify the HTTP_PROXY setting to ensure access to GitHub.

To ensure you can download pip packages successfully, you can modify the PIP_INDEX setting to use a mirror or alternative PyPI source. 

You can modify the BASE_IMAGE setting to use an alternative base image.

```bash
# Choose based on your CUDA version requirements:
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.6-cudnn9.5
```

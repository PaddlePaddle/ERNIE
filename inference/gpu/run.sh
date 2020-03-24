set -x
(($# != 2)) && echo "${0} data model" && exit -1

export LD_LIBRARY_PATH=fluid_inference/third_party/install/mkldnn/lib:fluid_inference/third_party/install/mklml/lib:fluid_inference/paddle/lib/:/home/work/cuda-9.0/lib64/:/home/work/cudnn/cudnn_v7_3_1_cuda9.0/lib64/:$LD_LIBRARY_PATH

./build/inference --logtostderr \
    --model_dir $2 \
    --data $1 \
    --repeat 5 \
    --output_prediction true \
    --use_gpu true \
    --device 0 \

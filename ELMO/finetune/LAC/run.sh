export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0 
export FLAGS_fast_eager_deletion_mode=1
export CUDA_VISIBLE_DEVICES=4

python train.py \
--traindata_dir data/train \
--model_save_dir model \
--use_gpu 1 \
--corpus_type_list train \
--corpus_proportion_list 1 \
--num_iterations 200000 \
--testdata_dir data/dev $@ \

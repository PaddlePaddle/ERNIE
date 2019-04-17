export CUDA_VISIBLE_DEVICES=0 
python  train.py \
--train_path='data/train/sentence_file_*'  \
--test_path='data/dev/sentence_file_*'  \
--vocab_path data/vocabulary_min5k.txt \
--learning_rate 0.2 \
--use_gpu True \
--all_train_tokens 35479 \
--local True $@

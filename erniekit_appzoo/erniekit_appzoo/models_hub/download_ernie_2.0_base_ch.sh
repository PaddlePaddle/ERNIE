#!/usr/bin/env bash
model_files_path="./ernie_2.0_base_ch_dir"
if [ ! -d $model_files_path ]; then
	mkdir $model_files_path
fi

#get pretrained ernie2.0 model params
wget -q --no-check-certificate http://bj.bcebos.com/wenxin-models/ernie_2.0_base_ch_open.tgz
tar xzf ernie_2.0_base_ch_open.tgz -C $model_files_path
rm ernie_2.0_base_ch_open.tgz
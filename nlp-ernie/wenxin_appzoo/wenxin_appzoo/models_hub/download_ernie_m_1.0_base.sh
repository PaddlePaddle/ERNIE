#!/usr/bin/env bash
model_files_path="./ernie_m_1.0_base_dir"
if [ ! -d $model_files_path ]; then
	mkdir $model_files_path
fi

#get pretrained ernie_m 1.0 model params
wget -q --no-check-certificate http://bj.bcebos.com/wenxin-models/ernie_m_1.0_base.tgz

tar xzf ernie_m_1.0_base.tgz -C $model_files_path
rm ernie_m_1.0_base.tgz
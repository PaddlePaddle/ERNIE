#!/usr/bin/env bash
model_files_path="../tools/data/data_aug/"
wget -q --no-check-certificate http://bj.bcebos.com/wenxin-models/vec2.txt
mv vec2.txt $model_files_path
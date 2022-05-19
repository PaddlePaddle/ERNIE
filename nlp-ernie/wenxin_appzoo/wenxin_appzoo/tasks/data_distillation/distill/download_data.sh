#!/usr/bin/env bash

#get distill data
wget --no-check-certificate http://bj.bcebos.com/wenxin-models/data_distillation_demo_data.tgz
tar xzf data_distillation_demo_data.tgz
mv ./distill/* .
rm -rf distill
rm data_distillation_demo_data.tgz
#!/usr/bin/env bash
if [[ ! -d output ]]; then
    mkdir -p output/
fi

cp wenxin_appzoo output/ -r

rm -fr output/wenxin_appzoo/tasks/plugin_demo/

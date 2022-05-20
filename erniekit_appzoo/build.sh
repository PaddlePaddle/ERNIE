#!/usr/bin/env bash
if [[ ! -d output ]]; then
    mkdir -p output/
fi

cp erniekit_appzoo output/ -r

rm -fr output/erniekit_appzoo/tasks/plugin_demo/

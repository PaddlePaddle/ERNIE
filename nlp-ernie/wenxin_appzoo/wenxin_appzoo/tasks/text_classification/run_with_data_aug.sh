#!/bin/bash
set -x
cd ../../models_hub/
sh download_data_aug.sh
cd ../tools/data/data_aug
python data_aug.py ../../../tasks/text_classification/data/train_data ../../../tasks/text_classification/data/train_data_aug
cd ../../../tasks/text_classification/data/train_data_aug
for file in ./*; do
    shuf $file -o $file
done
cd ../../
python run_trainer.py --param_path ./examples/cls_ernie_fc_ch_with_data_aug.json
exit 0
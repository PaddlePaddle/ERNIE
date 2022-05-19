remove_demo_data=$1
if [ $remove_demo_data -eq 1 ];then
    rm -r train_data test_data dev_data predict_data dict
fi
data_files_path="./"
#get data aug pretrained ernie model
wget --no-check-certificate http://bj.bcebos.com/wenxin-models/public_data/text_classification_data.tar.gz && \
tar xzvf text_classification_data.tar.gz -C $data_files_path && \
rm text_classification_data.tar.gz && \
mkdir dict && \
mv vocab.txt ./dict

remove_demo_data=$1
if [ $remove_demo_data -eq 1 ];then
    rm -r train_*
    rm -r dev_*
    rm -r test_*
    rm -r predict_*
    rm -r dict
fi
data_files_path="./"
#get data aug pretrained ernie model
wget --no-check-certificate http://bj.bcebos.com/wenxin-models/public_data/text_matching_data.tar.gz
tar xzvf text_matching_data.tar.gz -C $data_files_path
rm text_matching_data.tar.gz

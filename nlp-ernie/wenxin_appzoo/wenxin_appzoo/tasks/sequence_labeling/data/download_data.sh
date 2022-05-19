remove_demo_data=$1
if [ $remove_demo_data -eq 1 ];then
    rm -r *_data
fi
data_files_path="./"

wget --no-check-certificate http://bj.bcebos.com/wenxin-models/public_data/sequence_labeling_data.tar.gz
tar xzvf sequence_labeling_data.tar.gz -C $data_files_path
rm sequence_labeling_data.tar.gz

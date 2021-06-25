data_name=SST-2
data_tar=${data_name}.tar.gz
bos_url=https://unimo.bj.bcebos.com/data/SST-2.tar.gz

rm -rf $data_name
wget --no-check-certificate -q $bos_url
if [[ $? -ne 0 ]]; then
    echo "url link: $bos_url"
    echo "download data failed"
    exit 1
fi
tar zxf $data_tar
rm -f $data_tar
exit 0

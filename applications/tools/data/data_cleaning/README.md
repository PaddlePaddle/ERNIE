# 编码识别及转换工具

文心中的所有文件要求必须是utf-8格式，包括所有的样本集、词表文件、label_map等映射文件。 文心提供了文件格式转换的工具，位置在./applications/tools/data/data_cleaning
- 识别输入文本的编码格式:

```plain
# -i 指定输入文件路径
python file_encoding.py -i input_file
```

  - 将utf8格式的文本转成gb18030，如果输入文本不是utf8格式的，直接报错返回：

```plain
python file_encoding.py -i input_file -o output_file --utf8_to_gb18030
# 或者
python file_encoding.py -i input_file -o output_file -u2g
```

- 将gb18030的文本转成utf8，如果输入文本不是gb18030格式的，直接报错返回：

```plain
python file_encoding.py -i input_file -o output_file --gb18030_to_utf8
# 或者
python file_encoding.py -i input_file -o output_file -g2u
```
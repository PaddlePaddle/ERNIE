# 分词工具与词表生成工具

## 分词

开发套件用于分词的python脚本位于目录./applications/tools/data/wordseg/wordseg_lac.py：

- 文心中集成了[LAC](https://www.paddlepaddle.org.cn/modelbasedetail/lac)分词工具，运行该脚本前，需要先下载分词工具[lac](https://github.com/baidu/lac)包，或者直接通过pip命令安装：

```shell
pip install lac
```

- 运行脚本 wordseg_lac.py：

```
 python wordseg_lac.py -i "输入文件夹的目录" -o "输出文件的目录"
```

- wordseg_lac.py脚本的传参说明：

```shell
shell输入：
    python wordseg_lac.py -h
shell输出：
    optional arguments:
      -h, --help            show this help message and exit. 
      -i INPUT, --input INPUT                                # 分词目录，如果目录下有多个文件，则依次将目录下所有文件分词
      -o OUTPUT, --output OUTPUT                             # 指定分词结果文件保存的目录，分词后文件名为{原文件名_seg}后缀
      -c COLUMN_NUMBER, --column_number COLUMN_NUMBER        # 对指定列进行分词，如有多列使用逗号分割，默认为第1列
```

## 构建词表

如果用户想使用自己的样本集生成词表，则可使用上一节中的分词工具进行分词，得到分词后的样本文件后，直接使用词表生成工具，指定目录，生成自己的词表。词表生成工具位于tools/data/word_seg/build_voc.py，使用方式如下：

- 运行脚本build_voc.py

```shell
python build_voc.py -i "分好词的数据集目录路径" -o "生成的词表路径"
```

- build_voc.py脚本的传参说明：

```
输入：
    python build_voc.py -h
输出：
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
      -o OUTPUT, --output OUTPUT
      -sep SEPERATOR, --seperator SEPERATOR
      -c COLUMN_NUMBER, --column_number COLUMN_NUMBER
      -thr FEQ_THRESHOLD, --feq_threshold FEQ_THRESHOLD
      -ew EXTRA_WORDS [EXTRA_WORDS ...], --extra_words EXTRA_WORDS [EXTRA_WORDS ...]
      -sw STOP_WORDS [STOP_WORDS ...], --stop_words STOP_WORDS [STOP_WORDS ...]
```

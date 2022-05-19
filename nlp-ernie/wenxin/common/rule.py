# -*- coding: utf-8 -*
"""
some rule
"""


class MaxTruncation(object):
    """MaxTruncation：超长截断规则
    """
    KEEP_HEAD = 0  # 从头开始到最大长度截断
    KEEP_TAIL = 1  # 从头开始到max_len-1的位置截断，末尾补上最后一个id（词或字）
    KEEP_BOTH_HEAD_TAIL = 2  # 保留头和尾两个位置，然后按keep_head方式截断


class EmbeddingType(object):
    """EmbeddingType:文本数据需要转换的embedding类型：no_emb , ernie_emb
    """
    NONE_EMBEDDING = 0  # 不需要emb
    ERNIE_EMBEDDING = 1  # 用ernie生成emb
    FLUID_EMBEDDING = 2  # 使用fluid的op生成emb


class FluidDataType(object):
    """ FluidDataType data struct wrapper """

    def __init__(self, shape, dtype, lod_level, name=None):
        self.shape = shape
        self.dtype = dtype
        self.lod_level = lod_level
        self.name = name


class WordPieceType(object):
    """字词混合切分模式下，每个token的type"""
    SINGLE_TOKEN = 0  # 单个字
    WORD_START = 1  # 词首字符
    WORD_INCLUDE = 2  # 词中间字符


class DataShape(object):
    """DataShape:输入的数据类型
    """
    STRING = "string"  # string
    INT = "int"  # int64
    FLOAT = "float"  # float32


class InstanceName(object):
    """InstanceName:一些常用的命名
    """
    RECORD_ID = "id"
    RECORD_EMB = "emb"
    SRC_IDS = "src_ids"
    WORDSEG_IDS = "wordseg_ids"
    MASK_IDS = "mask_ids"
    LOSS_MASK = "loss_mask"
    SEQ_LENS = "seq_lens"
    SENTENCE_IDS = "sent_ids"
    POS_IDS = "pos_ids"
    TASK_IDS = "task_ids"
    PHONETIC_A_IDS = "phonetic_a_ids"
    PHONETIC_B_IDS = "phonetic_b_ids"
    GLYPH_A_IDS = "glyph_a_ids"
    GLYPH_B_IDS = "glyph_b_ids"
    GLYPH_C_IDS = "glyph_c_ids"
    GLYPH_D_IDS = "glyph_d_ids"



    REL_POS_IDS="rel_pos_ids"
    DEEP_IDS = "deep_ids"
    BEG_IDS = "beg_ids"
    END_IDS = "end_ids"

    #生成训练相关key
    TGT_LABEL = "tgt_label"
    TGT_POS = "tgt_pos"
    #生成解码相关key
    TGT_SRC_IDS = "tgt_src_ids"
    TGT_POS_IDS = "tgt_pos_ids"
    INIT_SCORES = "init_scores"
    PARENT_IDX = "parent_idx"
    TGT_MASK_IDS = 'tgt_mask_ids'
    DATA_IDS = 'data_ids'
    #多轮对话相关key
    ROLE_IDS = "role_ids"
    TURN_IDS = "turn_ids"
    TGT_PHONETIC_A_IDS = "tgt_phonetic_a_ids"
    TGT_PHONETIC_B_IDS = "tgt_phonetic_b_ids"
    TGT_GLYPH_A_IDS = "tgt_glyph_a_ids"
    TGT_GLYPH_B_IDS = "tgt_glyph_b_ids"
    TGT_GLYPH_C_IDS = "tgt_glyph_c_ids"
    TGT_GLYPH_D_IDS = "tgt_glyph_d_ids"

    # seq2seq的label域相关key
    TRAIN_LABEL_SRC_IDS = "train_label_src_ids"
    TRAIN_LABEL_MASK_IDS = "train_label_mask_ids"
    TRAIN_LABEL_SEQ_LENS = "train_label_seq_lens"
    INFER_LABEL_SRC_IDS = "infer_label_src_ids"
    INFER_LABEL_MASK_IDS = "infer_label_mask_ids"
    INFER_LABEL_SEQ_LENS = "infer_label_seq_lens"

    # term rank 相关的key
    TERM_POS = "term_pos"
    TERM_TOKENS_NUMS = "term_tokens_nums"
    TERM_INDEX = "term_index"
    TERM_PAIRS = "term_pairs"
    TERM_DIFFS = "term_diffs"

    SEQUENCE_EMB = "sequence_output"  # 词级别的embedding
    POOLED_EMB = "pooled_output"  # 句子级别的embedding

    TARGET_FEED = "target_feed"  # 保存模型时需要的入参：表示模型预测时需要输入的变量,tensor 或者variable类型
    TARGET_FEED_NAMES = "target_feed_name"  # 保存模型时需要的入参：表示模型预测时需要输入的变量名称和顺序
    TARGET_PREDICTS = "target_predicts"  # 保存模型时需要的入参：表示预测时最终输出的结果
    PREDICT_RESULT = "predict_result"  # 训练过程中需要传递的预测结果
    STUDENT_PREDICT_RESULT = "student_predict_result"  # 训练过程中需要传递的预测结果
    TEACHER_PREDICT_RESULT = "teacher_predict_result"  # 训练过程中需要传递的预测结果
    LABEL = "label"  # label

    TEACHER_CE_LOSS = "teacher_ce_loss"
    STUDENT_CE_LOSS = "student_ce_loss"
    DISTILL_LOSS = "distill_loss"
    PRED_LOSS = "pred_loss"
    LOSS = "loss"  # loss
    # CRF_EMISSION = "crf_emission"  # crf_emission

    TRAINING = "training"  # 训练过程
    EVALUATE = "evaluate"  # 评估过程
    TEST = "test" # 测试过程
    SAVE_INFERENCE = "save_inference"  # 保存inference model的过程
    INFERENCE = "inference"  # 预测过程

    STEP = "steps"
    SPEED = "speed"
    TIME_COST = "time_cost"
    GPU_ID = "gpu_id"

    FILE_CHECKPOINTS = "checkpoints"
    FILE_INFERENCE_MODEL = "inference_model"

    TYPE_PY_READER = "py_reader"
    TYPE_DATA_LOADER = "data_loader"

    # ERNIE-VIL相关key
    IMAGE_PIXEL_IDS = "image_pixel_ids"
    IMAGE_POSITION = "image_position"
    IMAGE_TAG_IDS = "image_tag_ids"
    TEXT_INDEX = "text_index"
    IMAGE_INDEX = "image_index"
    POS_INDEX = "pos_index"

    # ERNIE-Layout相关key
    POS_2D_IDS = "pos_2d_ids"
    SEGMENT_IDS = "segment_ids"

    # DynaBERT相关key
    HIDDEN_LAYERS = "hidden_layers"
    LOGIT = "logit"
    
    # prompt相关key
    LABEL_MAP_IDS = "label_map_ids"
    LABEL_TEXT_IDS = "label_text_ids"
    BATCH_SIZE = "batch_size"
    MAX_SEQ_LEN = "max_seq_len"


class FieldLength(object):
    """一个field在序列化成field_id_list的时候，占的长度是多少
    """
    CUSTOM_TEXT_FIELD = 3
    ERNIE_TEXT_FIELD = 6
    SINGLE_SCALAR_FIELD = 1
    ARRAY_SCALAR_FIELD = 2
    BASIC_TEXT_FIELD = 2
    GENERATE_LABEL_FIELD = 6
    ERNIE_TERM_RANK_TEXT_FIELD = 9
    ERNIT_TERM_RANK_LABEL_FIELD = 4
    # ERNIE-VIL RELATED VARIABLES
    ERNIEVIL_IMAGE_PIXEL_FIELD = 1
    ERNIEVIL_IMAGE_TAGS_FIELD = 1
    # ERNIE-Layout RELATED VARIABLES
    ERNIE_LAYOUT_SEQLABEL_FIELD = 10

class FleetMode(object):
    """Fleet模式
    """
    NO_FLEET = "NO_FLEET"
    CPU_MODE = "CPU"
    GPU_MODE = "GPU"


class UploadModelType(object):
    """模型上传的类型"""
    UPLOAD_HDFS_IMMEDIATE = "immediate"  # 实时上传到HDFS
    UPLOAD_HDFS_LAST_TIME = "last_time"  # 训练结束之后由paddlecloud平台进行集中上传


class StoreModelType(object):
    """模型保存方式的类型"""
    STORE_HDFS = "hadoop"  # 保存到hadoop集群上
    STORE_IREPO = "irepo"  # 保存到irepo模型仓库中


class EncryptType(object):
    """模型加密的方式"""
    ENCRYPT_NONE = None  # 不加密
    ENCRYPT_MEMORY = "memory"  # 内存加密
    ENCRYPT_FILE = "file"  # 文件加密


class InferenceRetcode(object):
    """ 预测服务返回码 """
    RET_OK = 200
    LOAD_JSON_FAILED = 201
    MISSING_FIELD = 202


class GraphMode(object):
    """图模式
    """
    #动态图
    DYGRAPH = "dynamic"
    #静态图
    STATIC = "static"

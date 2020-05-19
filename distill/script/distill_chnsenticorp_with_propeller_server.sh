set -x
export PYTHONPATH=.:./ernie/:${PYTHONPATH:-}
output_dir=./output/distill
teacher_dir=${output_dir}/teacher
student_dir=${output_dir}/student

# 1. finetune teacher
CUDA_VISIBLE_DEVICES=0 \
python3 -u ./example/finetune_classifier.py  \
    --data_dir ${TASK_DATA_PATH}/distill/chnsenticorp/teacher \
    --warm_start_from ${MODEL_PATH}/params \
    --vocab_file ${MODEL_PATH}/vocab.txt \
    --max_seqlen 128 \
    --run_config '{
        "model_dir": "'${teacher_dir}'",
        "max_steps": '$((10 * 9600 / 32))',
        "save_steps": 100,
        "log_steps": 10,
        "max_ckpt": 1,
        "skip_steps": 0,
        "eval_steps": 100
    }' \
    --hparam ${MODEL_PATH}/ernie_config.json \
    --hparam '{ # model definition
		"sent_type_vocab_size": None,    # default term in official config
		"use_task_id": False,
        "task_id": 0,
	}' \
    --hparam '{ # learn
      "warmup_proportion":  0.1,
      "weight_decay": 0.01,
      "use_fp16": 0,
      "learning_rate": 0.00005,
      "num_label": 2,
      "batch_size": 32 
    }' 

(($?!=0)) && echo "Something goes wrong at Step 1, please check" && exit -1

# 2. start a prediction server
CUDA_VISIBLE_DEVICES=1 \
python3 -m propeller.tools.start_server -p 8113 -m ${teacher_dir}/best/inference/ &
echo $! > pid.server

sleep 10


#. 3. learn from teacher
export CUDA_VISIBLE_DEVICES=0 
python3 ./distill/distill_chnsentocorp_with_propeller_server.py \
    --data_dir ${TASK_DATA_PATH}/distill/chnsenticorp/student \
    --vocab_file ${TASK_DATA_PATH}/distill/chnsenticorp/student/vocab.txt \
    --teacher_vocab_file ${MODEL_PATH}/vocab.txt \
    --max_seqlen 128 \
    --teacher_max_seqlen 128 \
    --server_batch_size 64 \
    --teacher_host tcp://localhost:8113 \
    --num_coroutine 10 \
    --run_config '{
        "model_dir": "'${student_dir}'",
        "max_steps": '$((100 * 9600 / 100))',
        "save_steps": 1000,
        "log_steps": 10,
        "max_ckpt": 1,
        "skip_steps": 0,
        "eval_steps": 100
    }' \
    --hparam '{ # model definition
        "num_label": 2,
        "vocab_size": 35000,
        "emb_size": 128,
        "initializer_range": 0.02,
	}' \
    --hparam '{ # learn  					    
      "warmup_proportion":  0.1,
      "weight_decay": 0.00,
      "learning_rate": 1e-4,
      "batch_size": 100 
    }' 
(($?!=0)) && echo "Something goes wrong at Step 2, please check" && exit -1

ps -ef|grep 'propeller.tools.start_server' |awk '{print $2}'|xargs kill -9



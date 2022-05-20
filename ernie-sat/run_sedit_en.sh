# 纯英文的语音编辑
python sedit_inference_0520.py \
--task_name edit \
--model_name paddle_checkpoint_en \
--uid p243_new \
--new_str 'for that reason cover is impossible to be given.' \
--prefix ./prompt/dev/ \
--source_language english \
--target_language english \
--output_name task_edit_pred.wav \
--voc pwgan_aishell3 \
--voc_config download/pwg_aishell3_ckpt_0.5/default.yaml \
--voc_ckpt download/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
--voc_stat download/pwg_aishell3_ckpt_0.5/feats_stats.npy \
--am fastspeech2_ljspeech \
--am_config download/fastspeech2_nosil_ljspeech_ckpt_0.5/default.yaml \
--am_ckpt download/fastspeech2_nosil_ljspeech_ckpt_0.5/snapshot_iter_100000.pdz \
--am_stat download/fastspeech2_nosil_ljspeech_ckpt_0.5/speech_stats.npy \
--phones_dict download/fastspeech2_nosil_ljspeech_ckpt_0.5/phone_id_map.txt
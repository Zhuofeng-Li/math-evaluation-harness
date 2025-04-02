set -ex

PROMPT_TYPE="torl"
MODEL_NAME_OR_PATH="/home/zhuofeng/.cache/huggingface/hub/models--GAIR--ToRL-7B/snapshots/383a9b80b1487a76c51798796bba6c62c1c5d8b0"

OUTPUT_DIR=results/${MODEL_NAME_OR_PATH}
DATA_NAMES="aime24"
# DATA_NAMES="gsm8k,math500,minerva_math,olympiadbench,aime24,amc23"
SPLIT="test"
NUM_TEST_SAMPLE=-1


# single-gpu
CUDA_VISIBLE_DEVICES=4,5,6,7 TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --data_names ${DATA_NAMES} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 8 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --max_tokens_per_call 3072 \
    --use_vllm \
    --overwrite \
    2>&1 | tee tmp.log


# multi-gpu
# python3 scripts/run_eval_multi_gpus.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir $OUTPUT_DIR \
#     --data_names ${DATA_NAMES} \
#     --prompt_type "cot" \
#     --temperature 0 \
#     --use_vllm \
#     --save_outputs \
#     --available_gpus 0,1,2,3,4,5,6,7 \
#     --gpus_per_model 1 \
#     --overwrite

#! /usr/bin/python
export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_DIR=/private/model/meta-llama/alpaca-7b
OUT_DIR=/private/home/zhumingye/data/data/augmented_from_rso_4_e3_i1
mkdir -p $OUT_DIR
# torchrun --nproc_per_node 8 --master_port 7834 /private/home/zhumingye/code/RRHF/data_generation/response_gen.py \
#                         --base_model $MODEL_DIR \
#                         --data_path "/private/home/zhumingye/data/data/rm-static/data" \
#                         --out_path $OUT_DIR \
#                         --batch_size 8

# python /private/home/zhumingye/code/RRHF/data_generation/split_files.py $OUT_DIR $OUT_DIR
bash /private/home/zhumingye/code/RRHF/data_generation/scoring_responses.sh $OUT_DIR
# python /private/home/zhumingye/code/RRHF/data_generation/make_data.py $OUT_DIR

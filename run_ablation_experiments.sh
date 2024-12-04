#!/bin/bash

# 设置控制变量
DATASET="gbmlgg"
TASK_TYPE="survival"
MODEL_TYPE="hmfm"
GATING_NET="CosMLP"
MAX_EXPERT=4
PROJ_DIM=256
SEED=5
C_REG=0

# 基础路径设置
BASE_DIR="/data_20/yinwendong/AMFM/results_${DATASET}/${MODEL_TYPE}"  # 使用完整的绝对路径
DATA_ROOT="/data_20/yinwendong/MCAT-master/data"
DATE=$(date +%Y_%m_%d)

# 通用参数
COMMON_PARAMS="--model_type ${MODEL_TYPE} --apply_sig --dataset ${DATASET} --task_type ${TASK_TYPE} --mome_gating_network ${GATING_NET} --c_reg ${C_REG} --data_root_dir ${DATA_ROOT} --max_expert ${MAX_EXPERT} --seed ${SEED}"

# 执行第一个命令
python main.py ${COMMON_PARAMS} --results_dir ${BASE_DIR}/1foldcv_${GATING_NET}_${MODEL_TYPE}_No_CoAFusion_${TASK_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d4_BPE_s${SEED} --mome_ablation_expert_id 0 --mof_ablation_expert_id 0 > ${BASE_DIR}/${DATASET}_${TASK_TYPE}_${MODEL_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d1_BPE_No_CoAFusion_${DATE}.txt
wait $!

# 执行第二个命令
python main.py ${COMMON_PARAMS} --results_dir ${BASE_DIR}/1foldcv_${GATING_NET}_${MODEL_TYPE}_No_SNNFusion_${TASK_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d4_BPE_s${SEED} --mome_ablation_expert_id 1 --mof_ablation_expert_id 0 > ${BASE_DIR}/${DATASET}_${TASK_TYPE}_${MODEL_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d1_BPE_No_SNNFusion_${DATE}.txt
wait $!

# 执行第三个命令
python main.py ${COMMON_PARAMS} --results_dir ${BASE_DIR}/1foldcv_${GATING_NET}_${MODEL_TYPE}_No_MILFusion_${TASK_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d4_BPE_s${SEED} --mome_ablation_expert_id 2 --mof_ablation_expert_id 0 > ${BASE_DIR}/${DATASET}_${TASK_TYPE}_${MODEL_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d1_BPE_No_MILFusion_${DATE}.txt
wait $!

# 执行第四个命令
python main.py ${COMMON_PARAMS} --results_dir ${BASE_DIR}/1foldcv_${GATING_NET}_${MODEL_TYPE}_No_ZeroFusion_${TASK_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d4_BPE_s${SEED} --mome_ablation_expert_id 3 --mof_ablation_expert_id 0 > ${BASE_DIR}/${DATASET}_${TASK_TYPE}_${MODEL_TYPE}_train_max_${MAX_EXPERT}_pdim_${PROJ_DIM}_d1_BPE_No_ZeroFusion_${DATE}.txt
wait $!

#!/bin/bash

dataset="ucec"
seed=6

# 第一组命令 (CoA和SNN)
python main.py --model_type hmfm --apply_sig --results_dir ./results_${dataset}/hafm/1foldcv_Cos_HMFM_No_CoAFusion_survival_train_max_4_pdim_256_AE2_RP_d4_s${seed} --dataset ${dataset} --mome_gating_network CosMLP --c_reg 0 --data_root_dir /root/sspaas-fs/ --max_expert 4 --seed ${seed} --mome_ablation_expert_id 0 > ./results_${dataset}/hafm/survival_HMFM_train_max_4_pdim_256_AE2_RP_d4_NoCoA_s${seed}_2024_12_6.txt &
pid1=$!

python main.py --model_type hmfm --apply_sig --results_dir ./results_${dataset}/hafm/1foldcv_Cos_HMFM_No_SNNFusion_survival_train_max_4_pdim_256_AE2_RP_d4_s${seed} --dataset ${dataset} --mome_gating_network CosMLP --c_reg 0 --data_root_dir /root/sspaas-fs/ --max_expert 4 --seed ${seed} --mome_ablation_expert_id 1 > ./results_${dataset}/hafm/survival_HMFM_train_max_4_pdim_256_AE2_RP_d4_NoSNN_s${seed}_2024_12_6.txt &
pid2=$!

# 等待第一组命令完全执行完
wait $pid1 $pid2

# 第二组命令 (MIL和Zero)
python main.py --model_type hmfm --apply_sig --results_dir ./results_${dataset}/hafm/1foldcv_Cos_HMFM_No_MILFusion_survival_train_max_4_pdim_256_AE2_RP_d4_s${seed} --dataset ${dataset} --mome_gating_network CosMLP --c_reg 0 --data_root_dir /root/sspaas-fs/ --max_expert 4 --seed ${seed} --mome_ablation_expert_id 2 > ./results_${dataset}/hafm/survival_HMFM_train_max_4_pdim_256_AE2_RP_d4_NoMIL_s${seed}_2024_12_6.txt &
pid3=$!

python main.py --model_type hmfm --apply_sig --results_dir ./results_${dataset}/hafm/1foldcv_Cos_HMFM_No_ZeroFusion_survival_train_max_4_pdim_256_AE2_RP_d4_s${seed} --dataset ${dataset} --mome_gating_network CosMLP --c_reg 0 --data_root_dir /root/sspaas-fs/ --max_expert 4 --seed ${seed} --mome_ablation_expert_id 3 > ./results_${dataset}/hafm/survival_HMFM_train_max_4_pdim_256_AE2_RP_d4_NoZero_s${seed}_2024_12_6.txt &
pid4=$!

# 等待第二组命令完全执行完
wait $pid3 $pid4
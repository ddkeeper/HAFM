# run baselines models
DATASET="brca"  #  gbmlgg, luad, blca
DATA_ROOT="/data_20/yinwendong/MCAT-master/data/"  
DATE=$(date +%Y_%m_%d)
SEED=6  # 
FOLDS=5  # 

# 1. SNN
echo "Starting SNN training..."
python main.py --model_type snn --mode omic --apply_sig --results_dir ./results_${DATASET}/snn/${FOLDS}foldcv_snn_survival_train/ --data_root_dir ${DATA_ROOT} --dataset ${DATASET} --seed ${SEED} > ./results_${DATASET}/snn/${DATASET}_survival_snn_${DATE}.txt
wait

# 2. AttnMIL
echo "Starting AttnMIL training..."
python main.py --model_type amil --mode path --apply_sig --results_dir ./results_${DATASET}/amil/${FOLDS}foldcv_amil_survival_train/ --fusion None --data_root_dir ${DATA_ROOT} --dataset ${DATASET} --seed ${SEED} > ./results_${DATASET}/amil/${DATASET}_survival_amil_${DATE}.txt
wait

# 3. MCAT
echo "Starting MCAT training..."
python main.py --model_type mcat --apply_sig --results_dir ./results_${DATASET}/mcat/${FOLDS}foldcv_mcat_survival_train/ --data_root_dir ${DATA_ROOT} --dataset ${DATASET} --seed ${SEED} > ./results_${DATASET}/mcat/${DATASET}_survival_mcat_${DATE}.txt
wait

# 4. MOTCAT
echo "Starting MOTCAT training..."
python main.py --model_type motcat --apply_sig --results_dir ./results_${DATASET}/motcat/${FOLDS}foldcv_motcat_survival_train/ --bs_micro 256 --data_root_dir ${DATA_ROOT} --dataset ${DATASET} --seed ${SEED} > ./results_${DATASET}/motcat/${DATASET}_survival_motcat_${DATE}.txt&
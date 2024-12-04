import torch
import numpy as np
import os
import time
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, precision_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
def test_model(args, model_path, test_loader, task_type):
    # 创建模型
    if args.model_type == 'mome':
        from models.model_mome import MoMETransformer
        model_dict = {
            'omic_sizes': args.omic_sizes, 
            'n_classes': args.n_classes, 
            'n_bottlenecks': args.n_bottlenecks,
            'gating_network': args.gating_network,
            'expert_idx': args.expert_idx,
            'ablation_expert_id': args.ablation_expert_id
        }
        model = MoMETransformer(**model_dict)
    else:
        raise NotImplementedError("Unsupported model type")

    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # 初始化结果存储
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_grades_true = []
    all_grades_pred = []
    inference_times = []

    with torch.no_grad():
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(test_loader):
            data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
            data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
            data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
            data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
            data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
            data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            c = c.type(torch.FloatTensor).cuda()
            grade = grade.type(torch.LongTensor).cuda()
            grade = grade - 2

            # 计算单次推理时间
            start_time = time.time()
            index_chunk_list = split_chunk_list(data_WSI, args.bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                hazards, S, Y_hat, A, hazard_grade, _, _ = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                break  # 只计算一个子批次的时间
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # 计算完整样本的结果
            hazard_grade_sum = None
            S_sum = None
            cnt = 0
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                hazards, S, Y_hat, A, hazard_grade, _, _ = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                if hazard_grade_sum is None:
                    hazard_grade_sum = hazard_grade
                    S_sum = S
                else:
                    hazard_grade_sum += hazard_grade
                    S_sum += S
                cnt += 1

            hazard_grade_avg = hazard_grade_sum / cnt
            S_avg = S_sum / cnt

            if task_type == 'survival':
                risk = -torch.sum(S_avg, dim=1).detach().cpu().numpy()
                all_risk_scores.extend(risk)
                all_censorships.extend(c.cpu().numpy())
                all_event_times.extend(event_time.numpy())
            elif task_type == 'grade':
                all_grades_true.extend(grade.cpu().numpy())
                all_grades_pred.extend(torch.softmax(hazard_grade_avg, dim=1).cpu().numpy())

    # 计算评估指标
    if task_type == 'survival':
        c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
        print(f"Test C-index: {c_index:.4f}")
        result = c_index
    elif task_type == 'grade':
        all_grades_true = np.array(all_grades_true)
        all_grades_pred = np.array(all_grades_pred)
        grade_true_bin = label_binarize(all_grades_true, classes=[0, 1, 2])
        micro_auc = roc_auc_score(grade_true_bin, all_grades_pred, multi_class='ovr', average='micro')
        micro_ap = average_precision_score(grade_true_bin, all_grades_pred, average='micro')
        micro_f1 = f1_score(all_grades_true, np.argmax(all_grades_pred, axis=1), average='micro')
        acc = np.mean(np.argmax(all_grades_pred, axis=1) == all_grades_true)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Micro AUC: {micro_auc:.4f}")
        print(f"Test Micro AP: {micro_ap:.4f}")
        print(f"Test Micro F1: {micro_f1:.4f}")
        result = (acc, micro_auc, micro_ap, micro_f1)

    # 计算平均推理时间
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")

    return result, avg_inference_time

def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list


import pandas as pd
import os
import torch

def find_file_by_slide_id(directory_path, case_id):
    # 构建完整的目录路径
    dir_path = os.path.join(directory_path, '')
    # 遍历目录中的文件
    for filename in os.listdir(dir_path):
        # 检查文件名中是否包含case_id
        if case_id in filename:
            return os.path.join(directory_path, filename)  # 返回找到的第一个文件名
    
    return None  # 如果没有找到文件，返回None

# 读取CSV文件
csv_path = "/home/yinwendong/CLAM_NEW/dataset_csv/tcga_blca.csv"  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_path)

# 遍历CSV文件中的每个case_id
normal_count = 0
abnormal_count = 0

for case_id in df['case_id']:
    directory_path = "/home/yinwendong/MCAT-master/data/tcga_blca_20x_features/pt_files/"
    wsi_path = find_file_by_slide_id(directory_path, case_id)
    if wsi_path:
        try:
            path_features = torch.load(wsi_path, map_location=torch.device('cpu'))
            normal_count += 1
            #print(f"正常加载文件：{wsi_path}")
            #print(f"path_features: {path_features.shape}")
        except RuntimeError as e:
            if "PytorchStreamReader failed reading zip archive" in str(e):
                abnormal_count += 1
                print(f"加载文件 {wsi_path} 时出现异常，对应的case_id为: {case_id}")
            else:
                raise  # 如果是其他类型的RuntimeError，则重新抛出异常
    else:
        abnormal_count += 1
        print(f"未找到case_id为{case_id}的文件")

print(f"正常加载的case_id数量：{normal_count}")
print(f"异常的case_id数量：{abnormal_count}")
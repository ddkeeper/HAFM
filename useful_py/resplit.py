import pandas as pd
import glob
import os

def process_split_files(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 只处理splits_0.csv到splits_4.csv
    for i in range(5):
        file_path = os.path.join(source_dir, f'splits_{i}.csv')
        if not os.path.exists(file_path):
            continue
            
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 找到val列最后一个非空值的索引
        last_val_index = df['val'].last_valid_index()
        
        # 获取test列的所有非空值
        test_values = df['test'].dropna().tolist()
        
        # 在val列的最后一个非空值后添加test列的值（复制而不是移动）
        for i, value in enumerate(test_values):
            df.loc[last_val_index + 1 + i, 'val'] = value
        
        # 获取原文件名
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_dir, file_name)
        
        # 保存修改后的文件到目标目录
        df.to_csv(target_path, index=False)

if __name__ == "__main__":
    # 设置源目录和目标目录
    #source_dir = "/data_20/yinwendong/AMFM/splits/5foldcv/tcga_luad"
    source_dir = "/data_20/yinwendong/CLAM_NEW/splits/tcga_ucec/"
    target_dir = "/data_20/yinwendong/AMFM/splits/5foldcv_new/tcga_ucec"
    process_split_files(source_dir, target_dir)

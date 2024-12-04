import pandas as pd

# 读取原始CSV文件
info = pd.read_csv("/data_20/yinwendong/AMFM/datasets_csv_new/tcga_ucec_all_clean.csv")

# 将grade为1的值转换为2
info.loc[(info['grade'].notna()) & (info['grade'] == 1), 'grade'] = 2

# 保存修改后的数据到新的CSV文件
info.to_csv("/data_20/yinwendong/AMFM/datasets_csv_new/tcga_ucec_all_clean_three_class.csv", index=False)

print("数据已成功修改并保存到新文件。")
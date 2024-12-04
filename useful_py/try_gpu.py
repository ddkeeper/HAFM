'''
每隔5分钟测试显存容量是否大于给定值，如果大于则循环执行命令列表的多条命令，每条命令执行前都需进行可用显存检测，
未通过检测则继续测试循环
'''
import time
import subprocess
import os

def get_gpu_memory_available():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        memory_available = int(output.strip())
        return memory_available
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("无法获取GPU信息，请确保已安装NVIDIA驱动并且nvidia-smi可用。")
        return None

def check_and_execute_commands(threshold, commands):
    for i, command in enumerate(commands, 1):
        print(f"准备执行第 {i} 条命令...")
        while True:
            memory_available = get_gpu_memory_available()
            
            if memory_available is not None:
                print(f"当前GPU可用显存: {memory_available} MB")
                
                if memory_available > threshold:
                    print(f"当前可用显存 {memory_available} MB 大于阈值 {threshold} MB，执行命令...")
                    # 使用 os.system 来确保命令在 shell 环境中正确执行
                    os.system(command)
                    print(f"第 {i} 条命令已开始执行。")
                    time.sleep(30)
                    break  # 跳出内部循环，准备执行下一条命令
                else:
                    print(f"当前可用显存 {memory_available} MB 未超过阈值，继续监控...")
            
            # 等待10分钟后再次检查
            time.sleep(600)
    
    print("所有命令已执行完毕。")

# 使用示例
if __name__ == "__main__":
    threshold = 9000  # 显存阈值（MB）
    c1 = "nohup python main.py --model_type mome --apply_sig --n_bottlenecks 2 --results_dir ./results_gbmlgg/grade_task/1foldcv_ADFM_grade_val_TF/ --gating_network transformer --task_type grade > tcga_gbmlgg_grade_ADFM_2024_9_15_TF.txt&"
    c2 = "nohup python main.py --model_type mome --apply_sig --n_bottlenecks 2 --results_dir ./results_gbmlgg/grade_task/1foldcv_ADFM_grade_val_MLP/ --gating_network MLP --task_type grade > tcga_gbmlgg_grade_ADFM_2024_9_15_MLP.txt&"
    c3 = "nohup python main.py --model_type mome --apply_sig --n_bottlenecks 2 --results_dir ./results_gbmlgg/grade_task/1foldcv_ADFM_grade_val_CNN1/ --gating_network CNN --task_type grade > tcga_gbmlgg_grade_ADFM_2024_9_15_CNN1.txt&"
    commands = [c1, c2, c3]
    check_and_execute_commands(threshold, commands)

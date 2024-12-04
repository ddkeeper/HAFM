import torch
import time
from models.model_DeepAttnMISL import *
from models.model_CoAttention import CoAFusion
from models.model_mome import SNNFusion, DropX2Fusion

from models.model_mof import *
import torch
import torch.nn.functional as F

def estimate_computation_time(model, x1, x2, num_iterations=100):
    has_params = any(p.requires_grad for p in model.parameters())
    if has_params:
        optimizer = torch.optim.Adam(model.parameters())
    total_time = 0
    target = torch.randn(x1.shape[0], x1.shape[2]).cuda()  # 创建与x1的第一维和第三维相同的随机目标张量，并移至GPU
    for _ in range(num_iterations):
        start_time = time.time()
        
        # 前向传播
        output, _ = model(x1, x2)
        #print(f"output: {output}")
        # 使用均方误差损失，假设输出与x1形状相同
        loss = F.mse_loss(output, target)
        
        # 反向传播
        if has_params:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        total_time += (end_time - start_time)
    
    return total_time / num_iterations


# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_dict_mome = {
        'SNNFusion': SNNFusion(dim=512).cuda(),
        'DropX2Fusion': DropX2Fusion(dim=512).cuda(),
        'CoAFusion': CoAFusion(dim=512).cuda(),
        'DAMISLFusion': DAMISLFusion(input_dim=512).cuda(),
        #'TMILFusion': TMILFusion(dim=512).cuda(),
        #'DAMISLFusion1': DAMISLFusion1(input_dim=512).cuda(),
        #'DAMISLFusion2': DAMISLFusion(input_dim=512).cuda(),
    }
    
    expert_dict_mof = {
        "DEQ": DEQ(dim=512).cuda(),
        "SA": SelfAttention(dim=512).cuda(),
        "OmicOnly": OmicOnly(dim=512).cuda(),
        "PathOnly": PathOnly(dim=512).cuda(),
    }

    n1, n2 = 10, 6  # 假设x1有10个序列,x2有6个序列
    dim = 512
    x1 = torch.randn(1, n1, dim).cuda()
    x2 = torch.randn(1, n2, dim).cuda()
    
    for name, model in expert_dict_mof.items():
        avg_time = estimate_computation_time(model, x1, x2)
        print(f"{name}: 平均每次迭代时间 {avg_time*1000:.2f} 毫秒")
from thop import profile
import torch
from models.model_DeepAttnMISL import *
from models.model_CoAttention import *
from models.model_mome import *

def calculate_expert_flops(expert_dict, n1, n2, dim=512):
    expert_flops = {}
    
    # 创建示例输入
    x1 = torch.randn(1, n1, dim)
    x2 = torch.randn(1, n2, dim)
    
    for name, expert in expert_dict.items():
        # 对于DropX2Fusion,我们只需要计算x1的FLOPs
        if name == 'DropX2Fusion':
            flops, _ = profile(expert, inputs=(x1, None))
        else:
            flops, _ = profile(expert, inputs=(x1, x2))
        expert_flops[name] = flops
    
    return expert_flops

# 使用示例
if __name__ == "__main__":
    expert_dict = {
        'SNNFusion': SNNFusion(dim=512),
        'DropX2Fusion': DropX2Fusion(dim=512),
        'CoAFusion': CoAFusion(dim=512),
        'DAMISLFusion': DAMISLFusion(input_dim=512)
    }
    
    n1, n2 = 10, 6  # 假设x1有10个序列,x2有6个序列
    flops = calculate_expert_flops(expert_dict, n1, n2)
    
    for name, flop in flops.items():
        print(f"{name}: {flop/1e6:.2f} MFLOPs")
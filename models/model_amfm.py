import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
import admin_torch
import sys

#import deqfusion
#from models.mm_model import *

#import DAMISLFusion
from models.model_DeepAttnMISL import *

#import CoAFusion
from models.model_CoAttention import *

#import Gating Network
from models.model_Gating import *

#import mixture of fusion for the input to the final prediction layer 
#from models.model_mof import *

class SelfAttention(nn.Module):
    def __init__(self, dim=512):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.multi_layer = TransLayer(dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_path, x_omic):
        # 连接cls_token、path特征和omic特征
        h_multi = torch.cat([self.cls_token, x_path, x_omic], dim=1)
        
        # 应用自注意力
        h = self.multi_layer(h_multi)
        
        # 返回cls_token对应的输出作为融合结果
        return h[:, 0, :], None
        
class SNNFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1

# Multi-modal Cosine Mixture-of-Experts
class MCMoE(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256, max_experts=2, ablation_expert_id=None):
        super().__init__()
        # 初始化专家
        self.DAMISLFusion = DAMISLFusion(input_dim=dim)
        self.SNNFusion = SNNFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        self.CoAFusion = CoAFusion(dim=dim)
        #self.TMILFusion = TMILFusion(dim=dim)
        
        experts = [
            self.CoAFusion,
            self.SNNFusion,
            self.DAMISLFusion,
            #self.TMILFusion,
            self.DropX2Fusion,
        ]

        # 如果指定了要删除的专家ID,从列表中删除
        if ablation_expert_id is not None and 0 <= ablation_expert_id < len(experts):
            del experts[ablation_expert_id]

        # 重新构建字典,保证ID连续
        self.routing_dict = {i: expert for i, expert in enumerate(experts)}
        
        # 添加负载计数器
        self.register_buffer('expert_counts', torch.zeros(len(self.routing_dict)))
        self.register_buffer('total_samples', torch.zeros(1))
        #self.accumulation_steps = 32  #累积更新步数

        # 添加专家激活数量分布统计
        self.max_experts = max_experts
        self.register_buffer('expert_k_counts', torch.zeros(max_experts + 1))  # 0到max_experts的分布
        self.register_buffer('total_samples_k', torch.zeros(1))
        # 初始化门控网络
        self.routing_network = MM_CosineGate(
            branch_num=len(self.routing_dict),
            dim=dim,
            max_experts=max_experts
        )
        
        # 用于计算负载均衡损失
        self.num_experts = len(self.routing_dict)
        
    def _update_load_counts(self, logits):
        """更新专家负载计数
        Args:
            logits: shape [batch_size, num_experts] 的tensor
        """
        # 统计整个batch中每个专家被激活的次数
        self.expert_counts += (logits > 0).float().sum(dim=0)  # 按专家维度求和
        self.total_samples += logits.size(0)  # 增加整个batch的样本数
        #print(f"self.top_k: {self.top_k}")
        #print(f"logits: {logits}")

    def _update_k_distribution(self, top_k):
        """更新专家激活数量分布统计"""
        for k in range(self.max_experts + 1):
            self.expert_k_counts[k] += torch.sum(top_k == k)

        #print(f"top_k: {top_k}")
        self.total_samples_k += len(top_k)        

    def _compute_balance_loss(self):
        """计算累积的负载均衡损失"""
        # 计算累积的负载分布
        load = self.expert_counts / self.expert_counts.sum()
        
        # 计算损失
        load_loss = load.pow(2).sum() * self.num_experts
        
        # 重置计数器
        self.expert_counts.zero_()
        self.total_samples.zero_()
        
        return load_loss

    def get_gating_params(self):
        """获取门控网络的相似度矩阵和门限值"""
        return {
            'sim_matrix': self.routing_network.sim_matrix.data.clone(),
            'activation_gates': self.routing_network.gates.data.clone(),
            'expert_k_counts': self.expert_k_counts.clone(),
            'expert_counts': self.expert_counts.clone()
        }

    def forward(self, x1, x2):
        # 获取门控网络的输出
        logits, top_k = self.routing_network(x1, x2)  # logits: [1, num_experts]
        
        # 初始化输出
        outputs = torch.zeros_like(x1)
        
        # 找出被选中的专家（logits > 0的专家）
        selected_experts = torch.where(logits[0] > 0)[0]  # 因为batch_size=1，所以用logits[0]
        num_selected = selected_experts.size(0)  # 获取选中专家的数量
        
        # 只处理被选中的专家
        for expert_id in selected_experts:
            # 获取专家权重
            expert_weight = logits[0, expert_id].unsqueeze(-1)  # [1]
            # 计算专家输出并加权
            expert_output = self.routing_dict[expert_id.item()](x1, x2)
            outputs += expert_weight * expert_output
        outputs /= num_selected
        
        # 计算辅助损失
        self._update_load_counts(logits)

        # 更新专家激活数量分布
        self._update_k_distribution(top_k)
        balance_loss = 0
        
        return outputs, balance_loss, None

# MoE + self attention
class AMFMTransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,
                 gating_network='MLP', expert_idx=0, ablation_expert_id=None,
                 mof_gating_network='MLP', mof_expert_idx=0, mof_ablation_expert_id=None, max_experts = 2, route_mode = True):
        super(AMFMTransformer, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
        self.route_mode = route_mode

        # Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        # WSI FC Layer
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        # MoME Layers
        self.routing_network = gating_network
        if self.routing_network == 'CosMLP':
            self.MoME_genom_1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            self.MoME_patho_1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            self.MoME_genom_2 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            self.MoME_patho_2 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
        else:
            raise RuntimeError

        # MoF Layer
        #self.MoF = MoF(dim=size[2], RoutingNetwork=mof_gating_network, expert_idx=mof_expert_idx, ablation_expert_id=mof_ablation_expert_id)
        
        #SA Layer for final fusion
        self.SA = SelfAttention(dim=size[2])

        # Classifiers
        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier_grade = nn.Linear(size[2], 3)
        self.act_grad = nn.LogSoftmax(dim=1)

    def get_gating_params(self):
        if self.routing_network == 'CosMLP':
            genom_gp = self.MoME_genom.get_gating_params()
            patho_gp = self.MoME_patho.get_gating_params()
            return genom_gp, patho_gp
        return None, None
    
    def _compute_balance_loss(self):
        if self.routing_network == 'CosMLP':
            genom_loss = self.MoME_genom._compute_balance_loss()
            patho_loss = self.MoME_patho._compute_balance_loss()
            return genom_loss + patho_loss
        return 0.0

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # 特征提取
        h_path_bag = self.wsi_net(x_path)
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic)
        
        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        if self.routing_network == 'CosMLP':
            
            # 并行双向编码
            #h_path_1, cost_p, corresponding_net_id_patho = self.MoME_patho_1(h_path_bag, h_omic_bag)
            #h_omic_1, cost_g, corresponding_net_id_genom = self.MoME_genom_1(h_omic_bag, h_path_bag)
            #h_path_2, cost_p, corresponding_net_id_patho = self.MoME_patho_2(h_path_1, h_omic_1)
            #h_omic_2, cost_g, corresponding_net_id_genom = self.MoME_genom_2(h_omic_1, h_path_1)
            # 交替编码
            h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho_1(h_path_bag, h_omic_bag)
            h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom_1(h_omic_bag, h_path_new)
            h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho_2(h_path_new, h_omic_new)
            h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom_2(h_omic_new, h_path_new)
        else:
            h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho(h_path_bag, h_omic_bag, hard=True)
            h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom(h_omic_bag, h_path_bag, hard=True)

        total_time_cost = cost_p + cost_g

        # 使用MoF进行最终融合
        #h, jacobian_loss, corresponding_net_id_fuse = self.MoF(h_path_2, h_omic_2, hard=self.route_mode)
        #使用SA进行最终融合
        h, _ = self.SA(h_path_new, h_omic_new)
        jacobian_loss, corresponding_net_id_fuse = None, -1
        # 预测
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        hazards_grade = self.classifier_grade(h)
        hazards_grade = self.act_grad(hazards_grade)
        
        attention_scores = {}
        expert_choices = {
            "corresponding_net_id_patho": corresponding_net_id_patho,
            "corresponding_net_id_genom": corresponding_net_id_genom,
            "corresponding_net_id_fuse": corresponding_net_id_fuse
        }
        
        return hazards, S, Y_hat, attention_scores, hazards_grade, jacobian_loss, total_time_cost, expert_choices

'''
# 双T结构 + 交替编码AE + 模态角色编码RP
class AMFMTransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,
                 gating_network='MLP', expert_idx=0, ablation_expert_id=None,
                 mof_gating_network='MLP', mof_expert_idx=0, mof_ablation_expert_id=None, max_experts = 2, route_mode = True):
        super(AMFMTransformer, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
        self.route_mode = route_mode

        # Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        # WSI FC Layer
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        # MoME Layers
        self.routing_network = gating_network
        if self.routing_network == 'CosMLP':
            self.MoME_genom_1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            self.MoME_patho_1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            #self.MoME_genom_2 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            #self.MoME_patho_2 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2], ablation_expert_id=ablation_expert_id, max_experts=max_experts)
        else:
            self.MoME_genom = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)
            self.MoME_patho = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)

        # Path Transformer + Attention Head
        #path_encoder_layer = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        path_encoder_layer = NystromTransformerEncoderLayer(
            d_model=size[2], 
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            activation='relu'
        )
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        # Omic Transformer + Attention Head
        #omic_encoder_layer = nn.TransformerEncoderLayer(d_model=size[2], nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        omic_encoder_layer = NystromTransformerEncoderLayer(
            d_model=size[2],
            nhead=8, 
            dim_feedforward=512,
            dropout=dropout,
            activation='relu'
        )
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        # Fusion Layer
        self.mm = nn.Sequential(*[nn.Linear(size[2]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])

        # Classifiers
        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier_grade = nn.Linear(size[2], 3)
        self.act_grad = nn.LogSoftmax(dim=1)

    def get_gating_params(self):
        if self.routing_network == 'CosMLP':
            genom_gp = self.MoME_genom.get_gating_params()
            patho_gp = self.MoME_patho.get_gating_params()
            return genom_gp, patho_gp
        return None, None
    
    def _compute_balance_loss(self):
        if self.routing_network == 'CosMLP':
            genom_loss = self.MoME_genom._compute_balance_loss()
            patho_loss = self.MoME_patho._compute_balance_loss()
            return genom_loss + patho_loss
        return 0.0

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]

        # 特征提取
        h_path_bag = self.wsi_net(x_path)
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic)
        
        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        if self.routing_network == 'CosMLP':
            # 交替编码
            h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho_1(h_path_bag, h_omic_bag)
            h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom_1(h_omic_bag, h_path_new)
            #h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho_2(h_path_new, h_omic_new)
            #h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom_2(h_omic_new, h_path_new)
        else:
            h_path_new, cost_p, corresponding_net_id_patho = self.MoME_patho(h_path_bag, h_omic_bag, hard=True)
            h_omic_new, cost_g, corresponding_net_id_genom = self.MoME_genom(h_omic_bag, h_path_bag, hard=True)

        total_time_cost = cost_p + cost_g

        # 处理path特征
        h_path_new = h_path_new.transpose(0, 1)
        #print(f"shape of h_path_2: {h_path_2.shape}")
        h_path_trans = self.path_transformer(h_path_new)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()
        # Ensure tensors are 2D
        #if h_path.dim() == 1:
        #    h_path = h_path.unsqueeze(0)
        #print(f"shape of h_path: {h_path.shape}")
        # 处理omic特征
        h_omic_new = h_omic_new.transpose(0, 1)
        #print(f"shape of h_omic_2: {h_omic_2.shape}")
        h_omic_trans = self.omic_transformer(h_omic_new)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        #A_omic = A_omic.squeeze(0)
        #h_omic = h_omic.squeeze(0)  # 如果h_omic还有多余的维度，移除它   
        A_omic = torch.transpose(A_omic, 1, 0)

        #A_omic = A_omic.squeeze(0)
        #h_omic = h_omic.squeeze()  # 如果h_omic还有多余的维度，移除它
        #print(f"shape of h_omic: {h_omic.shape}")
        #print(f"shape of A_omic: {A_omic.shape}")
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()
        #print(f"shape of h_omic: {h_omic.shape}")
        # 特征融合
        h = self.mm(torch.cat([h_path, h_omic], dim=0))
        
        jacobian_loss, corresponding_net_id_fuse = None, -1
        
        # 预测
        #logits = self.classifier(h)
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        #分级任务
        hazards_grade = self.classifier_grade(h).unsqueeze(0)
        hazards_grade = self.act_grad(hazards_grade)
        
        attention_scores = {
            'path': A_path,
            'omic': A_omic
        }
        
        expert_choices = {
            "corresponding_net_id_patho": corresponding_net_id_patho,
            "corresponding_net_id_genom": corresponding_net_id_genom,
            "corresponding_net_id_fuse": corresponding_net_id_fuse
        }
        
        return hazards, S, Y_hat, attention_scores, hazards_grade, jacobian_loss, total_time_cost, expert_choices
'''
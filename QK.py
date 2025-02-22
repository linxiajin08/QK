import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsciousAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, capacity=5):
        super().__init__()
        # 自主性提示投影（前额叶驱动）
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        # 非自主性提示投影（感官特征）
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        # 价值评估网络
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        )
        # 注意力容量限制（工作记忆阈值）
        self.capacity = capacity  
        self.scale = hidden_dim ** 0.5

    def forward(self, sensory_input, conscious_bias):
        """处理多模态输入并分配注意力
        Args:
            sensory_input (Tensor): [N, D] 感官信号矩阵 (非自主)
            conscious_bias (Tensor): [D,] 意识偏置向量 (自主)
        Returns:
            attended (Tensor): [K, D] 筛选后的重要信息 (K <= capacity)
            allocation (Tensor): [N,] 注意力分配权重
            value_map (Tensor): [N,] 各元素价值评估值
        """
        # 生成自主性查询（增强式）
        q = self.query_proj(conscious_bias)  # [D'] 意图聚焦
        q = F.normalize(q, p=2, dim=-1) * 2.0  # 自主性增益

        # 提取非自主性键
        k = self.key_proj(sensory_input)     # [N, D']

        # 价值评估（稀缺性感知）
        value_scores = self.value_net(sensory_input).squeeze()  # [N]

        # 注意力权值计算
        attn_logits = (q @ k.T) / self.scale  # [N]
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # 融合价值评估与注意力聚焦
        allocation = attn_weights * value_scores  # [N]
        allocation = F.normalize(allocation, p=1, dim=-1)  # 归一化

        # 容量控制（硬性筛选）
        top_values, top_indices = torch.topk(allocation, self.capacity)
        attended = sensory_input[top_indices]

        return attended, allocation, value_scores

class DialogManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensory_buffer = []  # 原始输入缓存
        self.attention_unit = ConsciousAttention(768, 256)
        self.conscious_state = nn.Parameter(torch.randn(768))  # 可学习的意识状态

        # 自主性增强模块
        self.consciousness = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 768)
        )

    def process_input(self, input_stream):
        """实时处理输入流"""
        # 更新意识状态（自主性引导）
        self.conscious_state = self.consciousness(self.conscious_state)
        
        # 注意力分配
        selected, weights, values = self.attention_unit(
            input_stream, 
            self.conscious_state
        )
        
        # 可视化接口（返回热力图数据）
        attention_heatmap = {
            'tokens': input_stream,
            'allocation': weights.detach(),
            'values': values.detach()
        }
        
        return selected, attention_heatmap

# 使用示例
dialog_ai = DialogManager()
input_data = torch.randn(20, 768)  # 模拟20个输入特征

with torch.no_grad():
    focused_data, visualization = dialog_ai.process_input(input_data)
    print(f"筛选后数据维度: {focused_data.shape}")  # [capacity, D]

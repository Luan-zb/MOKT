import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    自注意力池化层，将 n 个 d 维向量聚合为一个 d 维向量。
    """
    def __init__(self, dim=512):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, N, D]
        # weights: [B, N, 1]
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        
        # 聚合: [B, 1, N] @ [B, N, D] -> [B, 1, D] -> [B, D]
        fused = torch.bmm(weights.transpose(1, 2), x).squeeze(1)
        return fused, weights

class MultiModelFusion(nn.Module):
    """
    多模型融合架构：
    1. 对齐层 (MLP Alignment): 映射不同模型的特征到公共 512 维空间。
    2. 注意力池化 (Attention Pooling): 融合多个对齐后的特征。
    3. 回归层 (Regression Head): 输出基因表达预测值 (5245 维)。
    """
    def __init__(self, input_dims, hidden_dim=512, out_dim=5245):
        super(MultiModelFusion, self).__init__()
        
        # 1. 独立对齐层
        self.aligners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for in_d in input_dims
        ])
        
        # 2. 注意力融合层
        self.fusion = AttentionPooling(dim=hidden_dim)
        
        # 3. 回归层
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, out_dim)
        )

    def forward(self, features_list):
        """
        features_list: List of Tensors, each [B, D_i]
        """
        aligned_features = []
        for i, feat in enumerate(features_list):
            aligned_features.append(self.aligners[i](feat))
        
        # 堆叠特征: [B, N_models, hidden_dim]
        stacked = torch.stack(aligned_features, dim=1)
        
        # 注意力融合
        v_fusion, attn_weights = self.fusion(stacked)
        
        # 回归输出
        output = self.regressor(v_fusion)
        
        return output, attn_weights

if __name__ == "__main__":
    # 测试代码
    # input_dims: CHIEF(768), Feather(512), madeleine(512), prism(1280)
    test_dims = [768, 512, 512, 1280]
    model = MultiModelFusion(input_dims=test_dims)
    
    dummy_inputs = [torch.randn(8, d) for d in test_dims]
    out, weights = model(dummy_inputs)
    
    print(f"输入 Batch: 8")
    print(f"输出维度: {out.shape}") # 应为 [8, 5245]
    print(f"权重维度: {weights.shape}") # 应为 [8, 4, 1]

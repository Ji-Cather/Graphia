import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DegreeLevelRatePredictor(nn.Module):
    def __init__(self, seq_len: int, k: int, hidden_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, condition_dim: int = 0):
        """
        基于Transformer的度序列预测模型，预测level和rate
        :param seq_len: 输入序列长度
        :param k: 最大level值
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比率
        :param condition_dim: 条件维度
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.k = k
        
        # 计算模型维度
        if condition_dim > 0:
            d_model = hidden_dim + condition_dim
        else:
            d_model = hidden_dim
            
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        # 序列嵌入层
        self.seq_embedding = nn.Linear(k + 1, hidden_dim)  # k+1是因为要包含rate
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 - 分别预测level和rate
        self.level_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, k + 1),  # k+1个类别
            nn.LogSoftmax(dim=-1)
        )
        
        self.rate_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()  # 输出范围在[0,1]之间
        )

    def forward(self, 
                x_quantile: torch.Tensor,  # shape (batch_size, seq_len, 2) [level, rate]
                x_condition: torch.Tensor = None,  # shape (batch_size, condition_dim)
                return_quantile: bool = False):
        """
        前向传播
        :param x_quantile: 输入序列张量, shape (batch_size, seq_len, 2) [level, rate]
        :param x_condition: 条件张量, shape (batch_size, condition_dim)
        :param return_quantile: 是否返回quantile分布
        :return: 预测的level和rate
        """
        batch_size = x_quantile.shape[0]
        
        # 处理输入数据
        levels = x_quantile[:, :, 0].long()  # (batch_size, seq_len)
        rates = x_quantile[:, :, 1]  # (batch_size, seq_len)
        
        # 创建one-hot编码的level矩阵
        level_matrix = torch.zeros((batch_size, self.seq_len, self.k + 1), 
                                 device=x_quantile.device)
        level_matrix.scatter_(-1, levels.unsqueeze(-1), 1)
        
        # 在level+1位置放置rate值
        rate_matrix = torch.zeros_like(level_matrix)
        for i in range(batch_size):
            for j in range(self.seq_len):
                level = levels[i, j].item()
                if level < self.k:  # 确保level+1在有效范围内
                    rate_matrix[i, j, level + 1] = rates[i, j]
        
        # 合并level和rate信息
        x = level_matrix + rate_matrix
        
        # 序列嵌入
        x = self.seq_embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        if x_condition is not None:
            # 将x_condition的最后一个维度重复seq_len次
            x_condition = x_condition.unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat((x, x_condition), dim=-1)  # (batch_size, seq_len, hidden_dim + condition_dim)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        # 只使用最后一个时间步的输出进行预测
        x = x[:, -1, :]  # (batch_size, hidden_dim)
        
        # 预测level和rate
        level_logits = self.level_predictor(x)  # (batch_size, k+1)
        rate = self.rate_predictor(x)  # (batch_size, 1)
        
        if return_quantile:
            return level_logits, rate
        return level_logits, rate 
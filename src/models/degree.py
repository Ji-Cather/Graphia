import torch
import torch.nn as nn
import math
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import wasserstein_distance


def log_prop(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-30
    logw = torch.log(x + eps)
    assert not torch.isnan(logw).any(), "logw contains NaN values"
    return logw



# 添加批量归一化损失函数
class BatchNormDistributionLoss(nn.Module):
    def __init__(self):
        """
        批量归一化分布损失函数
        """
        super().__init__()
        
    def forward(self, pred_degrees, target_degrees):
        """
        计算批量归一化分布损失
        :param pred_degrees: 预测的节点度数
        :param target_degrees: 目标节点度数
        :return: 损失值
        """
        # 对预测和目标度数进行归一化
        pred_mean = pred_degrees.mean()
        pred_std = pred_degrees.std() + 1e-6
        norm_pred = (pred_degrees - pred_mean) / pred_std
        
        target_mean = target_degrees.mean()
        target_std = target_degrees.std() + 1e-6
        norm_target = (target_degrees - target_mean) / target_std
        
        # 计算归一化后的分布差异
        # 使用均方误差
        mse_loss = torch.mean((norm_pred - norm_target) ** 2)
        
        # 使用分布统计量的差异
        mean_loss = torch.abs(pred_mean - target_mean)
        std_loss = torch.abs(pred_std - target_std)
        
        # 总损失
        total_loss = mse_loss + 0.1 * mean_loss + 0.1 * std_loss
        
        return total_loss

class DegreeQuantileConverter(nn.Module):
    def __init__(self, max_degree: int):
        """
        将连续的degree值转换为k个quantile的表示
        :param max_degree: int, 最大度数
        :param k: int, quantile的数量
        """
        super().__init__()

        # 按照2为指数，从小到大表示
        log_2_k = int(torch.log2(torch.tensor(max_degree))) + 1
        self.k = log_2_k + 1
        self.register_buffer('quantile_values', torch.tensor([0,*[2**i for i in range(log_2_k)]]).float())

        
    def transform(self, degrees: torch.Tensor) -> torch.Tensor:
        """
        将degrees转换为quantile表示
        :param degrees: torch.Tensor, shape (batch_size, seq_len, 1)
        :return: torch.Tensor, shape (batch_size, seq_len, k)
        """
        batch_size, seq_len, _ = degrees.shape
        result = torch.zeros(batch_size, seq_len, self.k, device=degrees.device)
        
        # 对每个degree值计算其对应的quantile分布
        quantile_values_expanded = self.quantile_values.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # 计算权重
        weights = torch.zeros(batch_size, seq_len, self.k, device=degrees.device)
        
        # 处理小于等于最大quantile值的情况
        for i in range(self.k - 1):
            # 计算每个degree在quantile区间的相对位置
            lower_bound = quantile_values_expanded[:, :, i]
            upper_bound = quantile_values_expanded[:, :, i+1]
            
            # 计算权重
            mask = (degrees.squeeze(-1) >= lower_bound) & (degrees.squeeze(-1) < upper_bound)
            pos = (degrees.squeeze(-1) - lower_bound) / (upper_bound - lower_bound + 1e-10)
            pos = torch.clamp(pos, 0, 1)
            
            # 分配权重
            weights[:, :, i] = (1 - pos) * mask.float()
            weights[:, :, i+1] += pos * mask.float()
        
        # 处理大于最大quantile值的情况
        max_mask = (degrees.squeeze(-1) >= self.quantile_values[-1]).unsqueeze(-1).expand(-1, -1, self.k)
        weights[max_mask] = 0
        weights[max_mask[:, :, -1]] = 1
        
        result = weights
                
        log_result = log_prop(result)
        return log_result
        
    def inverse_transform(self, log_weights: torch.Tensor) -> torch.Tensor:
        """
        将quantile权重转换回degree值
        :param log_weights: torch.Tensor, shape (batch_size, k)
        :return: torch.Tensor, shape (batch_size, 1)
        """
        # 计算加权和得到degree值
        weights = torch.exp(log_weights)
        degrees = torch.sum(weights * self.quantile_values, dim=-1, keepdim=True)
        return degrees

class DegreePredictor(nn.Module):
    def __init__(self, seq_len: int, max_degree: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, condition_dim: int = 0):
        """
        基于Transformer的度序列预测模型
        :param seq_len: 输入序列长度
        :param max_degree: 最大度数(输出分布的维度)
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比率
        :param k_quantiles: quantile的数量
        """
        super().__init__()
        
        if condition_dim > 0:
            self.hidden_dim = hidden_dim + condition_dim
        else:
            self.hidden_dim = hidden_dim
            
        self.seq_len = seq_len
        self.max_degree = max_degree
        
        # 初始化degree quantile转换器
        self.degree_converter = DegreeQuantileConverter(max_degree=max_degree)
        
        self.k_quantiles = self.degree_converter.k
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 序列嵌入层 - 现在输入是k_quantiles维的quantile向量
        self.seq_embedding = nn.Linear(self.k_quantiles, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层,预测下一时刻的度数分布
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.k_quantiles),
            nn.LogSoftmax(dim=-1)  # 输出log概率
        )

    def forward(self, 
                x_degree: torch.Tensor, # shape (batch_size, seq_len, 1)
                x_condition: torch.Tensor = None, # shape (batch_size, seq_len, condition_dim)
                return_quantile: bool = False):
                
        """
        前向传播
        :param x: 输入序列张量, shape (batch_size, seq_len, 1)
        :param x_condition: 条件张量, shape (batch_size, seq_len, condition_dim)
        :param return_quantile: 是否返回quantile分布
        :return: 如果return_quantile为True，返回quantile分布和degree值；否则只返回degree值
        """
        # 将degree转换为quantile表示
        x_quantile = self.degree_converter.transform(x_degree)
        assert torch.sum(self.degree_converter.inverse_transform(x_quantile) - x_degree) < 1e-2, "quantile reconstruction error is too large"
        
        # 序列嵌入
        x = self.seq_embedding(x_quantile)  # (batch_size, seq_len, hidden_dim)
        if x_condition is not None:
            x = torch.cat((x, x_condition), dim=-1) # (batch_size, seq_len, hidden_dim + condition_dim)
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        x = x[:,-1,:]
        # 预测下一时刻的度数分布（log概率）
        log_quantile_weights = self.output_layer(x)  # (batch_size, k_quantiles)
        
        
        # 将quantile权重转换回degree值
        degree_values = self.degree_converter.inverse_transform(log_quantile_weights)
        
        
        
        if return_quantile:
            return log_quantile_weights, degree_values
        return degree_values

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        位置编码层
        :param d_model: 模型维度
        :param dropout: dropout比率
        :param max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        添加位置编码
        :param x: 输入张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    



if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 加载图数据
    import pickle as pkl
    import random
    import networkx as nx
    from torch.utils.data import Dataset, DataLoader
    
    
    
    # 加载图数据
    nx_graphs = pkl.load(open(f"/data/jiarui_ji/graph-generation-EDGE/graphs/community.pkl", 'rb'))
    random.shuffle(nx_graphs)
    l = len(nx_graphs)
    train_nx_graphs = nx_graphs[:int(0.2*l)]
    eval_nx_graphs = nx_graphs[int(0.8*l):int(0.9*l)]
    test_nx_graphs = nx_graphs[int(0.9*l):] 
    
    # 创建数据集类
    class DegreeDataset(Dataset):
        def __init__(self, graphs, seq_len=1):
            self.graphs = graphs
            self.seq_len = seq_len
            self.degrees = []
            
            # 收集所有图的节点度数
            for graph in graphs:
                degrees = [d for n, d in graph.degree()]
                self.degrees.extend(degrees)
            
            # 转换为张量并复制最后一个维度
            self.degrees = torch.tensor(self.degrees, dtype=torch.float32).reshape(-1, 1, 1).repeat(1, seq_len, 1)
            
        def __len__(self):
            return len(self.degrees)
        
        def __getitem__(self, idx):
            return self.degrees[idx]
    
    # 创建数据集
    seq_len = 5  # 使用更长的序列长度
    train_dataset = DegreeDataset(train_nx_graphs, seq_len=seq_len)
    eval_dataset = DegreeDataset(eval_nx_graphs, seq_len=seq_len)
    test_dataset = DegreeDataset(test_nx_graphs, seq_len=seq_len)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取最大度数
    max_degree = max([max(d for n, d in graph.degree()) for graph in nx_graphs])
    
    # 创建模型
    model = DegreePredictor(seq_len=seq_len, max_degree=max_degree)
    
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    degree_criterion = nn.MSELoss()
    
    # 损失权重
    degree_weight = 0.5
    quantile_weight = 0.5
    
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"验证数据大小: {len(eval_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    print(f"最大度数: {max_degree}")
    print(f"使用设备: {device}")
    
    # 训练循环 - 使用因果推理方式训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, degrees in enumerate(train_loader):
            degrees = degrees.to(device)
            
            # 对于每个时间步，使用前面的序列预测下一个值
            batch_loss = 0
            for t in range(1, seq_len):
                # 使用前t个时间步预测第t+1个时间步
                input_seq = degrees[:, :t, :]
                target_degree = degrees[:, t, :]
                
                # 前向传播，获取quantile权重和重构的degree值
                log_quantile_weights, predicted_degrees = model(input_seq, return_quantile=True)
                
                # 计算原始degree的quantile表示
                target_quantile = model.degree_converter.transform(target_degree.unsqueeze(1))
                target_quantile_weights = torch.exp(target_quantile).squeeze(1)
                
                # 计算degree损失和quantile损失
                degree_loss = degree_criterion(predicted_degrees, target_degree)
                
                # 使用原始quantile权重作为目标
                quantile_loss = -torch.mean(torch.sum(target_quantile_weights * log_quantile_weights, dim=1))
                
                # 总损失
                step_loss = degree_weight * degree_loss + quantile_weight * quantile_loss
                batch_loss += step_loss
            
            # 平均每个时间步的损失
            loss = batch_loss / (seq_len - 1)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for degrees in eval_loader:
                degrees = degrees.to(device)
                
                # 验证时也使用因果推理方式
                batch_loss = 0
                for t in range(1, seq_len):
                    input_seq = degrees[:, :t, :]
                    target_degree = degrees[:, t, :]
                    
                    log_quantile_weights, predicted_degrees = model(input_seq, return_quantile=True)
                    target_quantile = model.degree_converter.transform(target_degree.unsqueeze(1))
                    target_quantile_weights = torch.exp(target_quantile).squeeze(1)
                    
                    degree_loss = degree_criterion(predicted_degrees, target_degree)
                    quantile_loss = -torch.mean(torch.sum(target_quantile_weights * log_quantile_weights, dim=1))
                    
                    step_loss = degree_weight * degree_loss + quantile_weight * quantile_loss
                    batch_loss += step_loss
                
                loss = batch_loss / (seq_len - 1)
                val_loss += loss.item()
        
        val_loss /= len(eval_loader)
        print(f"Validation Loss: {val_loss:.4f}")
    
    # 测试 - 使用seq-1的输入预测seq的输出
    model.eval()
    test_inputs = []
    test_targets = []
    predicted_degrees_list = []
    
    with torch.no_grad():
        for degrees in test_loader:
            degrees = degrees.to(device)
            
            # 使用前seq_len-1个时间步预测最后一个时间步
            input_seq = degrees[:, :-1, :]
            target_degree = degrees[:, -1, :]
            
            # 预测最后一个时间步的度数
            _, predicted_degrees = model(input_seq, return_quantile=True)
            
            test_inputs.append(input_seq.cpu())
            test_targets.append(target_degree.cpu())
            predicted_degrees_list.append(predicted_degrees.cpu())
    
    # 合并所有批次的结果
    test_inputs = torch.cat(test_inputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    predicted_degrees = torch.cat(predicted_degrees_list, dim=0)
    
    # 计算MSE和MAE
    degree_mse = torch.mean((test_targets - predicted_degrees) ** 2)
    degree_mae = torch.mean(torch.abs(test_targets - predicted_degrees))
    
    print(f"测试输入形状: {test_inputs.shape}")
    print(f"测试目标形状: {test_targets.shape}")
    print(f"预测输出形状: {predicted_degrees.shape}")
    print(f"Degree MSE: {degree_mse.item():.4f}")
    print(f"Degree MAE: {degree_mae.item():.4f}")
    
    # 打印一些具体的例子
    print("\n具体例子:")
    for i in range(min(5, len(test_targets))):
        print(f"输入序列: {test_inputs[i, :, 0].numpy()}")
        print(f"目标degree值: {test_targets[i, 0].item():.2f}")
        print(f"预测degree值: {predicted_degrees[i, 0].item():.2f}")
        print(f"误差: {abs(test_targets[i, 0].item() - predicted_degrees[i, 0].item()):.2f}")
        print("---")
    
    # 尝试使用训练好的模型进行自回归生成
    print("\n自回归生成示例:")
    # 从测试集中选择一个序列作为起点
    start_seq = test_inputs[0].to(device)
    generated_seq = start_seq.clone().unsqueeze(0)
    
    # 生成10个新的时间步
    for i in range(10):
        # 使用当前序列预测下一个值
        _, next_degree = model(generated_seq, return_quantile=True)
        
        # 将预测值添加到序列中
        next_degree = next_degree.view(1, 1, 1)
        generated_seq = torch.cat([generated_seq[:, 1:, :], next_degree], dim=1)
        
        print(f"时间步 {i+1}: 预测degree值 = {next_degree.item():.2f}")
    
    print(f"生成序列: {generated_seq[0, :, 0].cpu().numpy()}")

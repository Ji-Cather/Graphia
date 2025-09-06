import numpy as np
import torch
from torch import nn


def compute_quantiles_torch_batch(data, q_list):
    """
    使用PyTorch高效计算一维数据的分位数
    :param data: 原始数据，形状为[B]，即长度为B的一维数据
    :param q_list: 分位数列表，例如[0.25, 0.5, 0.75]
    :return: 分位数值，形状为[B, len(q_list)]
    """
    # 确保数据是PyTorch张量
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    # 获取设备信息
    device = data.device
    
    # 确保数据是一维的
    if len(data.shape) > 1:
        raise ValueError("输入数据应该是一维的，形状为[B]")
    
    # 计算每个分位数的索引
    indices = []
    for q in q_list:
        # 将分位数转换为索引
        index = q * (len(data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(data) - 1)
        fraction = index - lower_index
        indices.append((lower_index, upper_index, fraction))
    
    # 对数据进行排序
    sorted_data, _ = torch.sort(data)
    
    # 计算分位数
    quantiles = torch.zeros((len(data), len(q_list)), device=device)
    for i, (lower_idx, upper_idx, frac) in enumerate(indices):
        if lower_idx == upper_idx:
            quantiles[:, i] = sorted_data[lower_idx]
        else:
            quantiles[:, i] = sorted_data[lower_idx] * (1 - frac) + sorted_data[upper_idx] * frac
    
    return quantiles

class QuantileLoss(nn.Module):
    def __init__(self, q_list=[0.25, 0.5, 0.75], weights=None):
        """
        多分位数组合损失函数
        :param q_list: 分位数列表，例如[0.25, 0.5, 0.75]表示同时优化25%、50%和75%分位点
        :param weights: 各分位数的权重，如果为None则平均权重
        """
        super(QuantileLoss, self).__init__()
        self.q_list = q_list
        if weights is None:
            self.weights = [1.0 / len(q_list)] * len(q_list)
        else:
            assert len(weights) == len(q_list), "权重数量必须与分位数数量相同"
            self.weights = weights
        
    def forward(self, y_pred, y_true):
        """
        计算多分位数组合损失
        :param y_pred: 预测值，形状为[batch_size, num_quantiles]
        :param y_true: 真实值，形状为[batch_size]
        :return: 组合损失
        """
        # 计算真实值的分位数
        y_true_quantiles = compute_quantiles_torch_batch(y_true, self.q_list)
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        # 确保y_pred的维度与y_true_quantiles匹配
        if y_pred.shape[0] != len(self.q_list):
            raise ValueError(f"预测值应该有{len(self.q_list)}个分位数，但实际有{y_pred.shape[1]}个")
        
        # 计算每个分位数的损失
        total_loss = 0
        individual_losses = []
        
        for i, q in enumerate(self.q_list):
            error = y_true_quantiles[:, i] - y_pred[:, i]
            quantile_loss = torch.mean(torch.max((q - 1) * error, q * error))
            total_loss += self.weights[i] * quantile_loss
            individual_losses.append(quantile_loss.item())
            
        return total_loss, individual_losses
    
    def get_loss_stats(self):
        """
        获取损失计算统计信息
        """
        return {
            'quantile_loss_count': len(self.q_list),
            'quantile_loss_time': 0.0,  # 这里可以添加实际的时间统计
            'avg_quantile_loss_time': 0.0
        }
        
        

        
        
# 添加直方图匹配损失函数
class HistogramMatchingLoss(nn.Module):
    def __init__(self, num_bins=10, use_wasserstein=True, use_kl=True, use_mmd=True, kernel_type='gaussian', sigma=1.0):
        """
        基于直方图匹配的损失函数
        :param num_bins: 直方图的箱数
        :param use_wasserstein: 是否使用Wasserstein距离，否则使用KL散度
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_wasserstein = use_wasserstein
        self.use_kl = use_kl
        self.use_mmd = use_mmd
        self.kernel_type = kernel_type
        self.sigma = sigma
        
    def rbf_kernel(self, x, y):
        """
        计算高斯核
        :param x: 第一个样本
        :param y: 第二个样本
        :return: 核函数值
        """
        return torch.exp(-torch.sum((x - y) ** 2) / (2 * self.sigma ** 2))
    
    
    
    
    
    def mmd_distance(self, x, y):
        """
        计算MMD距离
        :param x: 第一个分布的样本
        :param y: 第二个分布的样本
        :return: MMD距离
        """
        # 确保x和y是张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        # 计算核矩阵
        xx = torch.zeros((x.shape[0], x.shape[0]))
        yy = torch.zeros((y.shape[0], y.shape[0]))
        xy = torch.zeros((x.shape[0], y.shape[0]))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                xx[i, j] = self.rbf_kernel(x[i], x[j])
                
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                yy[i, j] = self.rbf_kernel(y[i], y[j])
                
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                xy[i, j] = self.rbf_kernel(x[i], y[j])
                
         # 计算MMD距离
        mmd_squared = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=1e-10))  # 防止负数或数值不稳定
        return mmd
        
    def compute_histogram(self, degrees, num_bins=None):
        """
        计算度数直方图
        :param degrees: 节点度数张量
        :param num_bins: 直方图的箱数
        :return: 直方图分布
        """
        if num_bins is None:
            num_bins = self.num_bins
            
        # 将度数展平
        flat_degrees = degrees.flatten().cpu().detach().numpy()
        
        # 计算直方图
        hist, _ = np.histogram(flat_degrees, bins=int(num_bins), density=False)
        hist = hist[1:]
        
        # 归一化
        if np.sum(hist) > 0 :
            hist = hist / np.sum(hist)
            
        return hist
    
    def forward(self, pred_degrees, target_degrees, name = "degree"):
        """
        计算直方图匹配损失
        :param pred_degrees: 预测的节点度数
        :param target_degrees: 目标节点度数
        :return: 损失值
        """
        # 计算预测和目标直方图
        pred_hist = self.compute_histogram(pred_degrees)
        target_hist = self.compute_histogram(target_degrees)
        
        loss = {}
        
        if self.use_wasserstein:
            # 使用Wasserstein距离
            # 创建累积分布函数
            pred_cdf = np.cumsum(pred_hist)
            target_cdf = np.cumsum(target_hist)
            
            # 计算Wasserstein距离
            distance = np.sum(np.abs(pred_cdf - target_cdf)) / len(pred_cdf)
            loss[f"histogram_wasserstein_distance_{name}"] = torch.tensor(distance, device=pred_degrees.device)
            
        if self.use_kl:
            # 使用KL散度
            eps = 1e-10
            kl_div = np.sum(target_hist * np.log((target_hist + eps) / (pred_hist + eps)))
            loss[f"histogram_kl_divergence_{name}"] = torch.tensor(kl_div, device=pred_degrees.device)
            
        if self.use_mmd:
            mmd_distance = self.mmd_distance(pred_hist, target_hist)
            loss[f"histogram_mmd_distance_{name}"] = torch.tensor(mmd_distance, device=pred_degrees.device)
            
        return loss
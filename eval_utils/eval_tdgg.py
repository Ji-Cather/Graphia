# LLMGGen/eval_utils/eval_tdgg.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler

def load_and_process_retrieval_data(file_path, 
                                    exclude_models=None, 
                                    metrics = ['hit@100', 'recall@100']):
    """
    加载并处理 retrieval 数据 (merged_dst_retrival_matrix.csv)
    先按 Group 列进行分组，然后对同 dataset, model, Group 的数据进行分组并计算平均值
    """
    df = pd.read_csv(file_path)
    
    # 如果提供了要排除的模型列表，则过滤掉这些模型
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]
        print(f"🔍 从 retrieval 数据中排除了 {len(exclude_models)} 个模型: {exclude_models}")
    
    # 选择需要的指标 (使用 hit@100 相关指标)
    
    
    # 确保所需列存在
    required_columns = ['model', 'dataset', 'Group'] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失列: {missing_columns}")
    
    # 按 model, dataset, Group 分组并计算平均值
    grouped_df = df.groupby(['model', 'dataset', 'Group'])[metrics].mean().reset_index()
    
    # 透视表转换，使 Group 成为列
    pivot_df = grouped_df.pivot_table(index=['model', 'dataset'], 
                                      columns='Group', 
                                      values=metrics,
                                      aggfunc='mean')
    
    # 扁平化列名
    pivot_df.columns = [f'{metric}_{group}' for metric, group in pivot_df.columns]
    
    # 重置索引
    pivot_df = pivot_df.reset_index()
    
    # 重命名列以标识来源
    rename_dict = {col: f"retrieval_{col}" for col in pivot_df.columns if col not in ['model', 'dataset']}
    pivot_df.rename(columns=rename_dict, inplace=True)
    
    return pivot_df

def load_and_process_edge_data(file_path, exclude_models=None):
    """
    加载并处理 edge 数据 (merged_edge_matrix.csv)
    选择指定指标
    """
    df = pd.read_csv(file_path)
    
    # 如果提供了要排除的模型列表，则过滤掉这些模型
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]
        print(f"🔍 从 edge 数据中排除了 {len(exclude_models)} 个模型: {exclude_models}")
    
    # 选择需要的指标
    metrics = ['label_acc', 'ROUGE_L', 'BERTScore_F1']
    
    # 确保所需列存在
    required_columns = ['model', 'dataset'] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失列: {missing_columns}")
    
    # 按 model 和 dataset 分组并计算平均值
    grouped_df = df.groupby(['model', 'dataset'])[metrics].mean().reset_index()
    
    # 重命名列以标识来源
    rename_dict = {metric: f"edge_{metric}" for metric in metrics}
    grouped_df.rename(columns=rename_dict, inplace=True)
    
    return grouped_df

def merge_datasets(retrieval_df, edge_df):
    """
    合并两个数据集，只保留两个表都包含的 model 和 dataset 组合
    """
    # 使用 inner join 只保留两个表都有的 model-dataset 组合
    merged_df = pd.merge(retrieval_df, edge_df, on=['model', 'dataset'], how='inner')
    return merged_df

def normalize_metrics(df):
    """
    对指标进行归一化处理（按 dataset 分组进行归一化，仅用于计算分数，不修改原始数据）
    """
    # 复制数据框以避免修改原始数据
    normalized_df = df.copy()
    
    # 获取所有 retrieval 指标列
    retrieval_metrics = [col for col in df.columns if col.startswith('retrieval_') and col not in ['model', 'dataset']]
    
    # 获取所有 edge 指标列
    edge_metrics = [col for col in df.columns if col.startswith('edge_') and col not in ['model', 'dataset']]
    
    # 按 dataset 分组进行归一化
    for dataset in df['dataset'].unique():
        # 获取当前 dataset 的数据索引
        dataset_mask = df['dataset'] == dataset
        dataset_indices = df[dataset_mask].index
        
        # 对 retrieval 指标进行归一化 (这些指标越高越好)
        if retrieval_metrics:
            scaler = MinMaxScaler()
            normalized_values = scaler.fit_transform(df.loc[dataset_indices, retrieval_metrics].fillna(0))
            normalized_df.loc[dataset_indices, retrieval_metrics] = normalized_values
        
        # 对 edge 指标进行归一化 (这些指标越高越好)
        if edge_metrics:
            scaler = MinMaxScaler()
            normalized_values = scaler.fit_transform(df.loc[dataset_indices, edge_metrics].fillna(0))
            normalized_df.loc[dataset_indices, edge_metrics] = normalized_values
    
    return normalized_df

def calculate_tdgg_social_fidelity_score(original_df, normalized_df, weights=None):
    """
    计算 tdgg-social fidelity score
    默认权重为 retrieval: 0.5, edge: 0.5
    """
    if weights is None:
        # 默认权重
        weights = {
            'retrieval': 0.5,
            'edge': 0.5
        }
    
    # 创建结果DataFrame，包含原始数据
    result_df = original_df.copy()
    
    # 获取所有 retrieval 指标列
    retrieval_metrics = [col for col in normalized_df.columns if col.startswith('retrieval_') and col not in ['model', 'dataset']]
    
    # 获取所有 edge 指标列
    edge_metrics = [col for col in normalized_df.columns if col.startswith('edge_') and col not in ['model', 'dataset']]
    
    # 计算 retrieval 部分的平均得分（使用归一化后的值）
    if retrieval_metrics:
        result_df['retrieval_score'] = normalized_df[retrieval_metrics].mean(axis=1)
    else:
        result_df['retrieval_score'] = 0
    
    # 计算 edge 部分的平均得分（使用归一化后的值）
    if edge_metrics:
        result_df['edge_score'] = normalized_df[edge_metrics].mean(axis=1)
    else:
        result_df['edge_score'] = 0
    
    # 计算最终的 tdgg-social fidelity score
    result_df['tdgg_social_fidelity_score'] = (
        weights['retrieval'] * result_df['retrieval_score'] + 
        weights['edge'] * result_df['edge_score']
    )
    
    return result_df

def find_top_models_per_dataset(df):
    """
    找出每个数据集中 retrieval_score、edge_score 和 tdgg_social_fidelity_score 最高的模型
    """
    top_models = {}
    
    # 按数据集分组
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        # 找到每个指标的最高分模型
        top_retrieval = dataset_df.loc[dataset_df['retrieval_score'].idxmax()]
        top_edge = dataset_df.loc[dataset_df['edge_score'].idxmax()]
        top_fidelity = dataset_df.loc[dataset_df['tdgg_social_fidelity_score'].idxmax()]
        
        top_models[dataset] = {
            'retrieval': {
                'model': top_retrieval['model'],
                'score': top_retrieval['retrieval_score']
            },
            'edge': {
                'model': top_edge['model'],
                'score': top_edge['edge_score']
            },
            'fidelity': {
                'model': top_fidelity['model'],
                'score': top_fidelity['tdgg_social_fidelity_score']
            }
        }
    
    return top_models

def print_top_models(top_models):
    """
    打印每个数据集的顶级模型
    """
    print("\n" + "="*80)
    print("各数据集顶级模型分析结果")
    print("="*80)
    
    for dataset, models in top_models.items():
        print(f"\n数据集: {dataset}")
        print("-" * 50)
        print(f"  最高 Retrieval Score 模型: {models['retrieval']['model']} (得分: {models['retrieval']['score']:.4f})")
        print(f"  最高 Edge Score 模型: {models['edge']['model']} (得分: {models['edge']['score']:.4f})")
        print(f"  最高 Fidelity Score 模型: {models['fidelity']['model']} (得分: {models['fidelity']['score']:.4f})")



import re

# 假设 retrieval_df 和 edge_df 已经加载

# 对 retrieval_df 应用重命名规则
def rename_retrieval_model(model_name):
    if re.match(r'grpo_.*_LIKR_reward_query_.*', model_name):
        return 'LLMGGen-seq'
    elif model_name.startswith('grpo_'):
        return 'LLMGGen'
    return model_name



# 对 edge_df 应用重命名规则
def rename_edge_model(model_name):
    if re.match(r'grpo_.*_sotopia_edge_.*', model_name):
        return 'LLMGGen-seq'
    elif model_name.startswith('grpo_'):
        return 'LLMGGen'
    return model_name



def evaluate_tdgg_social_fidelity(
    retrieval_file_path="LLMGGen/reports/concat/merged_dst_retrival_matrix.csv",
    edge_file_path="LLMGGen/reports/concat/merged_edge_matrix.csv",
    output_file_path="LLMGGen/reports/tdgg_social_fidelity_scores.csv",
    exclude_models=None,
    weights=None
):
    """
    主函数：评估 tdgg-social fidelity
    """
    # 加载和处理数据，排除指定模型
    retrieval_df = load_and_process_retrieval_data(retrieval_file_path, exclude_models)
    retrieval_df['model'] = retrieval_df['model'].apply(rename_retrieval_model)
    edge_df = load_and_process_edge_data(edge_file_path, exclude_models)
    edge_df['model'] = edge_df['model'].apply(rename_edge_model)
    
    # 合并数据，只保留两个表都包含的 model 和 dataset 组合
    merged_df = merge_datasets(retrieval_df, edge_df)
    
    # 归一化指标（仅用于计算分数）
    normalized_df = normalize_metrics(merged_df)
    
    # 计算 tdgg-social fidelity score
    result_df = calculate_tdgg_social_fidelity_score(merged_df, normalized_df, weights)
    
    # 保存结果（包含原始指标值和计算分数）
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file_path, index=False)
    
    print(f"✅ 评估完成，结果已保存至: {output_file_path}")
    print(f"📊 总共评估了 {len(result_df)} 个 model-dataset 组合")
    print("📋 前5行结果:")
    print(result_df.head())
    
    # 找出每个数据集的顶级模型
    top_models = find_top_models_per_dataset(result_df)
    print_top_models(top_models)
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 tdgg-social fidelity score")
    parser.add_argument("--retrieval_file", type=str, 
                        default="LLMGGen/reports/concat/merged_dst_retrival_matrix_raw.csv",
                        help="retrieval 矩阵文件路径")
    parser.add_argument("--edge_file", type=str,
                        default="LLMGGen/reports/concat/merged_edge_matrix.csv",
                        help="edge 矩阵文件路径")
    parser.add_argument("--output_file", type=str,
                        default="LLMGGen/reports/tdgg_social_fidelity_scores.csv",
                        help="输出文件路径")
    parser.add_argument("--exclude_models", type=str, nargs='*',
                        help="要排除的模型列表，例如: --exclude_models model1 model2")
    parser.add_argument("--retrieval_weight", type=float, default=0.5,
                        help="retrieval 部分的权重 (默认: 0.5)")
    parser.add_argument("--edge_weight", type=float, default=0.5,
                        help="edge 部分的权重 (默认: 0.5)")
    
    args = parser.parse_args()
    
    # 设置权重
    weights = {
        'retrieval': args.retrieval_weight,
        'edge': args.edge_weight
    }
    
    # 执行评估
    evaluate_tdgg_social_fidelity(
        retrieval_file_path=args.retrieval_file,
        edge_file_path=args.edge_file,
        output_file_path=args.output_file,
        exclude_models=args.exclude_models,
        weights=weights
    )
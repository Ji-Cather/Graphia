import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler

def normalize_metrics_for_idgg(df):
    """
    对 IDGG 指标进行归一化处理（按数据集分别归一化）
    负向指标需要反向处理
    """
    # 复制数据框以避免修改原始数据
    normalized_df = df.copy()
    
    # 获取所有负向指标列 (需要反向)
    negative_metrics = [col for col in df.columns if col in [
        'graph_list_degree_mmd', 'graph_list_cluster_mmd', 'graph_list_spectra_mmd',
        'graph_macro_D', 'graph_macro_num_chambers_diff'
    ]]
    
    # 获取所有正向指标列
    positive_metrics = [col for col in df.columns if col in [
        'graph_edge_overlap', 'graph_macro_auc@100_hub'
    ]]
    
    # print(f"Negative metrics (to be reversed): {negative_metrics}")
    # print(f"Positive metrics: {positive_metrics}")
    
    # 按数据集分别进行归一化
    for dataset in df['dataset'].unique():
        print(f"\nProcessing dataset: {dataset}")
        # 获取当前数据集的数据索引
        dataset_mask = df['dataset'] == dataset
        dataset_indices = df[dataset_mask].index
        
        # 显示原始数据
        # print("Original data for this dataset:")
        metric_columns = [col for col in df.columns if col.startswith('graph_')]
        # print(df.loc[dataset_indices, ['model'] + metric_columns])
        
        # 对负向指标进行归一化并反向 (1 - x) (这些指标越低越好)
        if negative_metrics:
            scaler = MinMaxScaler()
            original_values = df.loc[dataset_indices, negative_metrics].fillna(0)
            normalized_values = scaler.fit_transform(original_values)
            # print("Normalized negative values before reversing:")
            # print(normalized_values)
            # 反向处理负向指标
            reversed_values = 1 - normalized_values
            normalized_df.loc[dataset_indices, negative_metrics] = reversed_values
            # print("Reversed negative values (final):")
            # print(reversed_values)
        
        # 对正向指标进行归一化 (这些指标越高越好)
        if positive_metrics:
            scaler = MinMaxScaler()
            original_values = df.loc[dataset_indices, positive_metrics].fillna(0)
            normalized_values = scaler.fit_transform(original_values)
            normalized_df.loc[dataset_indices, positive_metrics] = normalized_values
            print("Normalized positive metrics:")
            print(normalized_values)
    
    return normalized_df

def calculate_idgg_social_fidelity_scores(original_df, normalized_df, weights=None):
    """
    计算 IDGG social fidelity scores
    包括三个子分数:
    1. macro_structure_score: 宏观拟真拓扑结构指标 (degree_mmd, cluster_mmd, spectra_mmd, edge_overlap)
    2. macro_phenomenon_score: 宏观现象拟合指标 (D, auc@100_hub, num_chambers_diff)
    3. idgg_social_fidelity_score: 综合分数
    """
    if weights is None:
        # 默认权重
        weights = {
            'macro_structure': 0.4,
            'macro_phenomenon': 0.6
        }
    
    # 创建结果DataFrame，包含原始数据
    result_df = original_df.copy()
    
    # 宏观拟真拓扑结构指标 (负向指标，已反向处理)
    macro_structure_metrics = [col for col in normalized_df.columns if col in [
        'graph_list_degree_mmd', 'graph_list_cluster_mmd', 'graph_list_spectra_mmd',
        'graph_edge_overlap'
    ]]
    
    # 宏观现象拟合指标 (负向指标，已反向处理)
    macro_phenomenon_metrics = [col for col in normalized_df.columns if col in [
        'graph_macro_D', 'graph_macro_auc@100_hub', 'graph_macro_num_chambers_diff'
    ]]
    
    print(f"Macro structure metrics: {macro_structure_metrics}")
    print(f"Macro phenomenon metrics: {macro_phenomenon_metrics}")
    
    # 计算宏观拟真拓扑结构得分（使用归一化后的值）
    if macro_structure_metrics:
        result_df['macro_structure_score'] = normalized_df[macro_structure_metrics].mean(axis=1)
    else:
        result_df['macro_structure_score'] = 0
    
    # 计算宏观现象拟合得分（使用归一化后的值）
    if macro_phenomenon_metrics:
        result_df['macro_phenomenon_score'] = normalized_df[macro_phenomenon_metrics].mean(axis=1)
    else:
        result_df['macro_phenomenon_score'] = 0
    
    # 计算最终的 idgg-social fidelity score
    result_df['idgg_social_fidelity_score'] = (
        weights['macro_structure'] * result_df['macro_structure_score'] + 
        weights['macro_phenomenon'] * result_df['macro_phenomenon_score']
    )
    
    return result_df

import re

def rename_retrieval_model(model_name):
    if re.match(r'grpo_.*_LIKR_reward_query_.*', model_name):
        return 'LLMGGen-seq'
    elif model_name.startswith('grpo_'):
        return 'LLMGGen'
    return model_name

def load_and_process_graph_list_data(file_path, exclude_models=None):
    """
    加载并处理 graph list 数据 (merged_graph_list_matrix.csv)
    处理指标: degree_mmd, cluster_mmd, spectra_mmd, D
    """
    df = pd.read_csv(file_path)
    
    # 如果提供了要排除的模型列表，则过滤掉这些模型
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]
        print(f"🔍 从 graph list 数据中排除了 {len(exclude_models)} 个模型: {exclude_models}")
    
    # 模型重命名逻辑
    # 对于满足grpo_的都rename为LLMGGen
    df['model'] = df['model'].apply(rename_retrieval_model)
    
    # 选择需要的指标
    metrics = ['degree_mmd', 'cluster_mmd', 'spectra_mmd']
    
    # 确保所需列存在
    required_columns = ['model', 'dataset'] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失列: {missing_columns}")
    
    # 按 model、dataset 分组并计算平均值
    grouped_df = df.groupby(['model', 'dataset'])[metrics].mean().reset_index()
    
    # 重命名列以标识来源
    rename_dict = {metric: f"graph_list_{metric}" for metric in metrics}
    grouped_df.rename(columns=rename_dict, inplace=True)
    
    return grouped_df

def load_and_process_graph_data(file_path, exclude_models=None):
    """
    加载并处理 graph 数据 (merged_graph_matrix.csv)
    处理指标: wedge_count, triangle_count, edge_overlap
    """
    df = pd.read_csv(file_path)
    
    # 如果提供了要排除的模型列表，则过滤掉这些模型
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]
        print(f"🔍 从 graph 数据中排除了 {len(exclude_models)} 个模型: {exclude_models}")
    
    # 模型重命名逻辑
    # 对于满足grpo_的都rename为LLMGGen
    df['model'] = df['model'].apply(rename_retrieval_model)
    
    # 选择需要的指标
    metrics = [ 'edge_overlap']
    
    # 确保所需列存在
    required_columns = ['model', 'dataset'] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失列: {missing_columns}")
    
    # 按 model、dataset 分组并计算平均值
    grouped_df = df.groupby(['model', 'dataset'])[metrics].mean().reset_index()
    
    # 重命名列以标识来源
    rename_dict = {metric: f"graph_{metric}" for metric in metrics}
    grouped_df.rename(columns=rename_dict, inplace=True)
    
    return grouped_df

def load_and_process_graph_macro_data(file_path, exclude_models=None):
    """
    加载并处理 graph macro 数据 (merged_graph_macro_matrix.csv)
    处理指标: num_chambers_diff, auc@100_hub, D
    """
    df = pd.read_csv(file_path)
    
    # 如果提供了要排除的模型列表，则过滤掉这些模型
    if exclude_models:
        df = df[~df['model'].isin(exclude_models)]
        print(f"🔍 从 graph macro 数据中排除了 {len(exclude_models)} 个模型: {exclude_models}")
    
    # 模型重命名逻辑
    # 对于满足grpo_的都rename为LLMGGen
    df['model'] = df['model'].apply(rename_retrieval_model)
    
    # 选择需要的指标
    metrics = ['num_chambers_diff', 'auc@100_hub', 'D']
    
    # 确保所需列存在
    required_columns = ['model', 'dataset'] + metrics
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失列: {missing_columns}")
    
    # 按 model、dataset 分组并计算平均值
    grouped_df = df.groupby(['model', 'dataset'])[metrics].mean().reset_index()
    
    # 重命名列以标识来源
    rename_dict = {metric: f"graph_macro_{metric}" for metric in metrics}
    grouped_df.rename(columns=rename_dict, inplace=True)
    
    return grouped_df

def merge_all_datasets(graph_list_df, graph_df, graph_macro_df):
    """
    合并所有数据集，只保留所有表都包含的 model、dataset 组合
    """
    # 合并三个数据框
    merged_df = pd.merge(graph_list_df, graph_df, on=['model', 'dataset'], how='inner')
    merged_df = pd.merge(merged_df, graph_macro_df, on=['model', 'dataset'], how='inner')
    
    return merged_df

def print_top_models_idgg(top_models):
    """
    打印每个数据集的顶级模型 (IDGG 版本)
    """
    print("\n" + "="*80)
    print("IDGG Social Fidelity 各数据集顶级模型分析结果")
    print("="*80)
    
    for dataset, models in top_models.items():
        print(f"\n数据集: {dataset}")
        print("-" * 50)
        print(f"  最高 Macro Structure Score 模型: {models['macro_structure']['model']} (得分: {models['macro_structure']['score']:.4f})")
        print(f"  最高 Macro Phenomenon Score 模型: {models['macro_phenomenon']['model']} (得分: {models['macro_phenomenon']['score']:.4f})")
        print(f"  最高 Fidelity Score 模型: {models['fidelity']['model']} (得分: {models['fidelity']['score']:.4f})")

def find_top_models_per_dataset_idgg(df):
    """
    找出每个数据集中三个 IDGG 指标的最高分模型
    """
    top_models = {}
    
    # 按数据集分组
    for dataset in df['dataset'].drop_duplicates().values:
        dataset_df = df[df['dataset'] == dataset]
        
        # 检查是否有数据
        if dataset_df.empty:
            continue
            
        # 找到每个指标的最高分模型
        top_structure = dataset_df.loc[dataset_df['macro_structure_score'].idxmax()]
        top_phenomenon = dataset_df.loc[dataset_df['macro_phenomenon_score'].idxmax()]
        top_fidelity = dataset_df.loc[dataset_df['idgg_social_fidelity_score'].idxmax()]
        
        top_models[dataset] = {
            'dataset': dataset,
            'macro_structure': {
                'model': top_structure['model'],
                'score': top_structure['macro_structure_score']
            },
            'macro_phenomenon': {
                'model': top_phenomenon['model'],
                'score': top_phenomenon['macro_phenomenon_score']
            },
            'fidelity': {
                'model': top_fidelity['model'],
                'score': top_fidelity['idgg_social_fidelity_score']
            }
        }
    
    return top_models

def evaluate_idgg_social_fidelity(
    graph_list_file_path="LLMGGen/reports/concat/merged_graph_list_matrix.csv",
    graph_file_path="LLMGGen/reports/concat/merged_graph_matrix.csv",
    graph_macro_file_path="LLMGGen/reports/concat/merged_graph_macro_matrix.csv",
    output_file_path="LLMGGen/reports/idgg_social_fidelity_scores.csv",
    exclude_models=None,
    weights=None
):
    """
    主函数：评估 IDGG social fidelity
    """
    # 加载和处理数据，排除指定模型
    graph_list_df = load_and_process_graph_list_data(graph_list_file_path, exclude_models)
    graph_df = load_and_process_graph_data(graph_file_path, exclude_models)
    graph_macro_df = load_and_process_graph_macro_data(graph_macro_file_path, exclude_models)
    
    # 合并数据，只保留所有表都包含的 model 和 dataset 组合
    merged_df = merge_all_datasets(graph_list_df, graph_df, graph_macro_df)
    
    # 检查合并后的数据
    print(f"合并后的数据形状: {merged_df.shape}")
    print("合并后的列:", merged_df.columns.tolist())
    
    # 归一化指标（仅用于计算分数）
    normalized_df = normalize_metrics_for_idgg(merged_df)
    
    # 显示归一化后的样本数据
    print("归一化后的样本数据:")
    print(normalized_df.head())
    
    # 计算 IDGG social fidelity scores
    result_df = calculate_idgg_social_fidelity_scores(merged_df, normalized_df, weights)
    
    # 显示计算后的样本数据
    print("计算分数后的样本数据:")
    print(result_df[['model', 'dataset', 'macro_structure_score', 'macro_phenomenon_score', 'idgg_social_fidelity_score']].head())
    
    # 保存结果（包含原始指标值和计算分数）
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存包含原始指标和计算分数的数据
    result_df.to_csv(output_file_path, index=False)
    
    print(f"✅ IDGG 评估完成，结果已保存至: {output_file_path}")
    print(f"📊 总共评估了 {len(result_df)} 个 model-dataset 组合")
    print("📋 前5行结果:")
    print(result_df.head())
    
    # 找出每个数据集的顶级模型
    top_models = find_top_models_per_dataset_idgg(result_df)
    print_top_models_idgg(top_models)
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 IDGG social fidelity score")
    parser.add_argument("--graph_list_file", type=str, 
                        default="LLMGGen/reports/concat/merged_graph_list_matrix.csv",
                        help="graph list 矩阵文件路径")
    parser.add_argument("--graph_file", type=str,
                        default="LLMGGen/reports/concat/merged_graph_matrix.csv",
                        help="graph 矩阵文件路径")
    parser.add_argument("--graph_macro_file", type=str,
                        default="LLMGGen/reports/concat/merged_graph_macro_matrix.csv",
                        help="graph macro 矩阵文件路径")
    parser.add_argument("--output_file", type=str,
                        default="LLMGGen/reports/idgg_social_fidelity_scores.csv",
                        help="输出文件路径")
    parser.add_argument("--exclude_models", type=str, nargs='*',
                        help="要排除的模型列表，例如: --exclude_models idgg_csv_processed_edge")
    parser.add_argument("--macro_structure_weight", type=float, default=0.6,
                        help="宏观结构部分的权重 (默认: 0.4)")
    parser.add_argument("--macro_phenomenon_weight", type=float, default=0.4,
                        help="宏观现象部分的权重 (默认: 0.6)")
    
    args = parser.parse_args()
    
    # 设置权重
    weights = {
        'macro_structure': args.macro_structure_weight,
        'macro_phenomenon': args.macro_phenomenon_weight
    }
    
    # 执行评估
    evaluate_idgg_social_fidelity(
        graph_list_file_path=args.graph_list_file,
        graph_file_path=args.graph_file,
        graph_macro_file_path=args.graph_macro_file,
        output_file_path=args.output_file,
        exclude_models=args.exclude_models,
        weights=weights
    )
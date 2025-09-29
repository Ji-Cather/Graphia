import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from math import pi
def plot_tdgg_radar(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                   output_dir="Graphia/reports/figures/",
                   figsize=(12, 10)):
    """
    绘制 tdgg-social fidelity scores 的雷达图
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft',
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }
    df['model'] = df['model'].replace(model_rename_map)
    
    # 检查必要的列是否存在
    missing_columns = [col for col in metrics if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 按数据集分组绘制雷达图
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制每个模型的数据
        colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_df)))
        
        for idx, (i, row) in enumerate(dataset_df.iterrows()):
            # 获取模型的三个维度分数
            values = row[metrics].tolist()
            values += values[:1]  # 闭合图形
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        
        # 设置标题和图例
        ax.set_title(f'TDGG Social Fidelity Scores - {dataset}', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 设置网格
        ax.grid(True)
        ax.set_ylim(0, 1)
        
        # 保存图片
        output_path = Path(output_dir) / f"tdgg_radar_{dataset}.pdf"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 已保存 {dataset} 数据集的雷达图至: {output_path}")

def plot_tdgg_radar_comparison(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                              output_dir="Graphia/reports/figures/",
                              figsize=(15, 12)):
    """
    绘制所有数据集的综合雷达图对比
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 获取所有唯一的模型
    all_models = df['model'].unique()
    
    # 创建子图
    n_datasets = len(df['dataset'].unique())
    cols = min(3, n_datasets)
    rows = (n_datasets + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw=dict(projection='polar'))
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_models)))
    model_color_map = dict(zip(all_models, colors))
    
    # 为每个数据集绘制雷达图
    for idx, dataset in enumerate(df['dataset'].unique()):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 绘制每个模型的数据
        for _, row in dataset_df.iterrows():
            # 获取模型的三个维度分数
            values = row[metrics].tolist()
            values += values[:1]  # 闭合图形
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['model'], color=model_color_map[row['model']])
            ax.fill(angles, values, alpha=0.25, color=model_color_map[row['model']])
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        
        # 设置标题
        ax.set_title(f'{dataset}', size=12, pad=20)
        
        # 设置网格和范围
        ax.grid(True)
        ax.set_ylim(0, 1)
    
    # 隐藏多余的子图
    for idx in range(n_datasets, len(axes)):
        fig.delaxes(axes[idx])
    
    # 添加图例
    handles, labels = axes[0].get_legend_handles_labels()
    # 只在第一个子图上显示图例，或者创建统一图例
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.1))
    
    # 保存图片
    output_path = Path(output_dir) / "tdgg_radar_comparison.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存综合雷达图对比至: {output_path}")

def plot_score_distributions(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                            output_dir="Graphia/reports/figures/"):
    """
    绘制各维度评分的分布图
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 为每个指标绘制分布图
    for idx, metric in enumerate(metrics):
        # 使用 seaborn 绘制箱线图
        sns.boxplot(data=df, x='dataset', y=metric, ax=axes[idx])
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "score_distributions.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 已保存评分分布图至: {output_path}")


def plot_formatted_overall_radar(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                                output_dir="Graphia/reports/figures/",
                                figsize=(12, 10)):
    """
    绘制格式化标签的整体雷达图，突出显示 Graphia 模型
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 检查必要的列是否存在
    missing_columns = [col for col in metrics if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 重命名模型
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft'
    }
    
    df['model'] = df['model'].replace(model_rename_map)
    
    # 按模型分组并计算平均值
    avg_scores = df.groupby('model')[metrics].mean().reset_index()
    
    # 格式化指标名称（首字母大写，下划线变空格）
    format_metric_map = {
        "selection_score": r"$S_\text{sel}$",
        "edge_score": r"$S_\text{edge}$",
        "tdgg_social_fidelity_score": r"$S_\text{TDGG}$"
    }
    formatted_metrics = [format_metric_map.get(metric, metric) for metric in metrics]
    # 定义模型绘制顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-R1-Distill-Qwen-32B',
        'Meta-Llama-3.1-70B-Instruct',
        'Graphia-seq',
        'Graphia'
    ]
    
    # 按照指定顺序重新排列数据
    ordered_data = []
    for model_name in model_order:
        model_data = avg_scores[avg_scores['model'] == model_name]
        if not model_data.empty:
            ordered_data.append(model_data)
    
    # 合并排序后的数据
    if ordered_data:
        reordered_scores = pd.concat(ordered_data, ignore_index=True)
    else:
        reordered_scores = avg_scores.copy()
    
    # 设置雷达图参数
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # 定义基于您提供配色方案的颜色映射
    color_map = {
        'Qwen3-8b': '#1f77b4',  # 深海蓝 Deep Ocean Blue - 主色，稳重可靠
        'Qwen3-8b-sft': '#ff7f0e',  # 珊瑚橙 Coral Orange - 高亮色，突出表现
        'Qwen3-32B': '#2ca02c',  # 森林绿 Forest Green - 自然系，表达增长
        'DeepSeek-R1-Distill-Qwen-32B': '#9467bd',  # 紫藤紫 Wisteria Purple - 优雅中性
        'Meta-Llama-3.1-70B-Instruct': '#f7b84d',  # 日落金 Sunset Gold - 明亮温暖
        'Graphia-seq': '#d62728',  # 洋红粉 Magenta Pink - 强烈个性，吸引注意
        'Graphia': '#17becf'  # 松石青 Turquoise Teal - 清新冷调
    }
    
    # 绘制每个模型的数据
    for idx, (_, row) in enumerate(reordered_scores.iterrows()):
        model_name = row['model']
        values = row[metrics].tolist()
        values += values[:1]
        
        # 获取颜色，如果未指定则使用默认颜色
        color = color_map.get(model_name, plt.cm.tab10(idx))
        
        # 设置线条属性
        if model_name in ['Graphia-seq', 'Graphia']:
            linewidth = 4
            alpha = 1.0
            zorder = 10  # 确保在最上层
            markersize = 10
        else:
            linewidth = 2.5
            alpha = 0.85
            zorder = 5
            markersize = 8
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
                label=model_name, color=color, markersize=markersize,
                alpha=alpha, zorder=zorder, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color=color, zorder=zorder-1)
    
    # 添加标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(formatted_metrics, fontsize=30, fontweight='bold')
    
    # 设置图表样式
    ax.set_ylim(0, 1)
    # 修改径向标签（0.2, 0.4等）的大小
    ax.tick_params(axis='y', labelsize=24)   # 设置径向标签字体大小
    ax.grid(True, alpha=0.3)
    
    # 设置标题
    # plt.title('Average TDGG Social Fidelity Scores\n(Across All Datasets)', 
    #           size=16, fontweight='bold', pad=30)
    
    # 添加图例（放在图表外部）
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncols = 2,
               fontsize=22, frameon=True, fancybox=True, shadow=True)
    
    # 美化网格
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f8f9fa')
    
    # 保存图片
    output_path = Path(output_dir) / "formatted_overall_tdgg_radar.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 已保存格式化整体雷达图至: {output_path}")
    
    return reordered_scores
def plot_formatted_dataset_radar(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                                output_dir="Graphia/reports/figures/",
                                figsize=(12, 10)):
    """
    为每个数据集绘制格式化标签的雷达图
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 检查必要的列是否存在
    missing_columns = [col for col in metrics if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 重命名模型
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft'
    }
    
    df['model'] = df['model'].replace(model_rename_map)
    
    # 定义模型绘制顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-R1-Distill-Qwen-32B',
        'Meta-Llama-3.1-70B-Instruct',
        'Graphia-seq',
        'Graphia'
    ]
    
    # 定义颜色映射
    color_map = {
        'Qwen3-8b': '#1f77b4',  # 深海蓝 Deep Ocean Blue - 主色，稳重可靠
        'Qwen3-8b-sft': '#ff7f0e',  # 珊瑚橙 Coral Orange - 高亮色，突出表现
        'Qwen3-32B': '#2ca02c',  # 森林绿 Forest Green - 自然系，表达增长
        'DeepSeek-R1-Distill-Qwen-32B': '#9467bd',  # 紫藤紫 Wisteria Purple - 优雅中性
        'Meta-Llama-3.1-70B-Instruct': '#f7b84d',  # 日落金 Sunset Gold - 明亮温暖
        'Graphia-seq': '#d62728',  # 洋红粉 Magenta Pink - 强烈个性，吸引注意
        'Graphia': '#17becf'  # 松石青 Turquoise Teal - 清新冷调
    }
    
    # 为每个数据集绘制雷达图
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 按模型分组并计算平均值
        avg_scores = dataset_df.groupby('model')[metrics].mean().reset_index()
        
        # 按照指定顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = avg_scores[avg_scores['model'] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            reordered_scores = pd.concat(ordered_data, ignore_index=True)
        else:
            reordered_scores = avg_scores.copy()
        
        # 设置雷达图参数
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        
        # 格式化指标名称（首字母大写，下划线变空格）
        format_metric_map = {
        "selection_score": r"$S_\text{sel}$",
        "edge_score": r"$S_\text{edge}$",
        "tdgg_social_fidelity_score": r"$S_\text{TDGG}$"
    }
        formatted_metrics = [format_metric_map.get(metric, metric) for metric in metrics]
        # 绘制每个模型的数据
        for idx, (_, row) in enumerate(reordered_scores.iterrows()):
            model_name = row['model']
            values = row[metrics].tolist()
            values += values[:1]
            
            # 获取颜色，如果未指定则使用默认颜色
            color = color_map.get(model_name, plt.cm.tab10(idx))
            
            # 设置线条属性
            if model_name in ['Graphia-seq', 'Graphia']:
                linewidth = 4
                alpha = 1.0
                zorder = 10  # 确保在最上层
                markersize = 10
            else:
                linewidth = 2.5
                alpha = 0.85
                zorder = 5
                markersize = 8
            
            ax.plot(angles, values, 'o-', linewidth=linewidth, 
                    label=model_name, color=color, markersize=markersize,
                    alpha=alpha, zorder=zorder, markeredgecolor='white', markeredgewidth=1.5)
            ax.fill(angles, values, alpha=0.15, color=color, zorder=zorder-1)
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_metrics, fontsize=26, fontweight='bold')
        
        # 设置图表样式
        ax.set_ylim(0, 1)
        # 修改径向标签（0.2, 0.4等）的大小
        ax.tick_params(axis='y', labelsize=12)  # 设置径向标签字体大小
        ax.grid(True, alpha=0.3)
        
        # 设置标题
        plt.title(f'Average TDGG Social Fidelity Scores\n({dataset})', 
                  size=16, fontweight='bold', pad=30)
        
        # 添加图例（放在图表外部）
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncols = 2,
                   fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        # 美化网格
        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('#f8f9fa')
        
        # 保存图片
        output_path = Path(output_dir) / f"formatted_radar_{dataset}.pdf"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 已保存 {dataset} 数据集的格式化雷达图至: {output_path}")

def plot_formatted_overall_radar(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                                output_dir="Graphia/reports/figures/",
                                figsize=(10, 8)):
    """
    绘制格式化标签的整体雷达图，突出显示 Graphia 模型
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 检查必要的列是否存在
    missing_columns = [col for col in metrics if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 重命名模型
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft',
        'DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Q-32B',
        'Meta-Llama-3.1-70B-Instruct': 'Llama3-70B'
    }

    df['model'] = df['model'].replace(model_rename_map)
    
    # 按模型分组并计算平均值
    avg_scores = df.groupby('model')[metrics].mean().reset_index()
    
    # 格式化指标名称（首字母大写，下划线变空格）
    format_metric_map = {
        "selection_score": r"$S_\text{sel}$",
        "edge_score": r"$S_\text{edge}$",
        "tdgg_social_fidelity_score": r"$S_\text{TDGG}$"
    }
    formatted_metrics = [format_metric_map.get(metric, metric) for metric in metrics]
    
    # 定义模型绘制顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-Q-32B',
        'Llama3-70B',
        'Graphia-seq',
        'Graphia'
    ]
    
    
    
    # 按照指定顺序重新排列数据
    ordered_data = []
    for model_name in model_order:
        model_data = avg_scores[avg_scores['model'] == model_name]
        if not model_data.empty:
            ordered_data.append(model_data)
    
    # 合并排序后的数据
    if ordered_data:
        reordered_scores = pd.concat(ordered_data, ignore_index=True)
    else:
        reordered_scores = avg_scores.copy()
    
    # 设置雷达图参数
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # 定义基于您提供配色方案的颜色映射
    color_map = {
        'Qwen3-8b': '#1f77b4',  # 深海蓝 Deep Ocean Blue - 主色，稳重可靠
        'Qwen3-8b-sft': '#ff7f0e',  # 珊瑚橙 Coral Orange - 高亮色，突出表现
        'Qwen3-32B': '#2ca02c',  # 森林绿 Forest Green - 自然系，表达增长
        'DeepSeek-R1-Distill-Qwen-32B': '#9467bd',  # 紫藤紫 Wisteria Purple - 优雅中性
        'Meta-Llama-3.1-70B-Instruct': '#f7b84d',  # 日落金 Sunset Gold - 明亮温暖
        'Graphia-seq': '#d62728',  # 洋红粉 Magenta Pink - 强烈个性，吸引注意
        'Graphia': '#17becf'  # 松石青 Turquoise Teal - 清新冷调
    }
    
    # 绘制每个模型的数据
    for idx, (_, row) in enumerate(reordered_scores.iterrows()):
        model_name = row['model']
        values = row[metrics].tolist()
        values += values[:1]
        
        # 获取颜色，如果未指定则使用默认颜色
        color = color_map.get(model_name, plt.cm.tab10(idx))
        
        # 设置线条属性
        if model_name in ['Graphia-seq', 'Graphia']:
            linewidth = 4
            alpha = 1.0
            zorder = 5 # 确保在最上层
            markersize = 10
        else:
            linewidth = 2.5
            alpha = 0.85
            zorder = 4
            markersize = 8
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
                label=model_name, color=color, markersize=markersize,
                alpha=alpha, zorder=zorder, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color=color, zorder=zorder-1)
    
    
    # 设置图表样式
    ax.set_ylim(0, 1)
    # 修改径向标签（0.2, 0.4等）的大小
    ax.tick_params(axis='y', labelsize=24,zorder=15)  # 设置径向标签字体大小
    ax.grid(True, alpha=0.3)
    
        # 添加标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(formatted_metrics, fontsize=26, fontweight='bold',zorder=20)

    # 设置标题
    # plt.title('Average TDGG Social Fidelity Scores\n(Across All Datasets)', 
    #           size=16, fontweight='bold', pad=30)
    
    # 添加图例（放在图表外部）
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncols = 2,
               fontsize=22, frameon=True, fancybox=True, shadow=True)
    
    # 美化网格
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#f8f9fa')
    
    # 保存图片
    output_path = Path(output_dir) / "formatted_overall_tdgg_radar.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 已保存格式化整体雷达图至: {output_path}")
    
    return reordered_scores


def plot_formatted_dataset_radar_combined(scores_file_path="Graphia/reports/tdgg_social_fidelity_scores.csv",
                                        output_dir="Graphia/reports/figures/",
                                        figsize=(20, 15)):
    """
    在一个大图中为所有数据集绘制格式化标签的雷达图
    
    Parameters:
    scores_file_path (str): tdgg_social_fidelity_scores.csv 文件路径
    output_dir (str): 图片输出目录
    figsize (tuple): 图片大小
    """
    
    # 读取评分数据
    df = pd.read_csv(scores_file_path)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取需要绘制的指标
    metrics = ['selection_score', 'edge_score', 'tdgg_social_fidelity_score']
    
    # 检查必要的列是否存在
    missing_columns = [col for col in metrics if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 重命名模型
    model_rename_map = {
        'qwen3': 'Qwen3-8b',
        'qwen3_sft': 'Qwen3-8b-sft'
    }
    
    df['model'] = df['model'].replace(model_rename_map)
    
    # 重命名数据集
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',

    }
    
    df['dataset'] = df['dataset'].replace(dataset_rename_map)
    
    # 定义模型绘制顺序
    model_order = [
        'Qwen3-8b',
        'Qwen3-8b-sft', 
        'Qwen3-32B',
        'DeepSeek-R1-Distill-Qwen-32B',
        'Meta-Llama-3.1-70B-Instruct',
        'Graphia-seq',
        'Graphia'
    ]
    
    # 定义颜色映射
    color_map = {
        'Qwen3-8b': '#1f77b4',  # 深海蓝 Deep Ocean Blue - 主色，稳重可靠
        'Qwen3-8b-sft': '#ff7f0e',  # 珊瑚橙 Coral Orange - 高亮色，突出表现
        'Qwen3-32B': '#2ca02c',  # 森林绿 Forest Green - 自然系，表达增长
        'DeepSeek-R1-Distill-Qwen-32B': '#9467bd',  # 紫藤紫 Wisteria Purple - 优雅中性
        'Meta-Llama-3.1-70B-Instruct': '#f7b84d',  # 日落金 Sunset Gold - 明亮温暖
        'Graphia-seq': '#d62728',  # 洋红粉 Magenta Pink - 强烈个性，吸引注意
        'Graphia': '#17becf'  # 松石青 Turquoise Teal - 清新冷调
    }
    
    # 获取所有数据集
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    
    # 计算子图布局
    cols = min(2, n_datasets)
    rows = (n_datasets + cols - 1) // cols
    
    # 创建大图
    fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # 处理单个子图的情况
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 格式化指标名称（首字母大写，下划线变空格）
    # formatted_metrics = [metric.replace('_', ' ').title() for metric in metrics]
    format_metric_map = {
        "selection_score": r"$S_\text{sel}$",
        "edge_score": r"$S_\text{edge}$",
        "tdgg_social_fidelity_score": r"$S_\text{TDGG}$"
    }
    formatted_metrics = [format_metric_map.get(metric, metric) for metric in metrics]


    # 为每个数据集绘制雷达图
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # 按模型分组并计算平均值
        avg_scores = dataset_df.groupby('model')[metrics].mean().reset_index()
        
        # 按照指定顺序重新排列数据
        ordered_data = []
        for model_name in model_order:
            model_data = avg_scores[avg_scores['model'] == model_name]
            if not model_data.empty:
                ordered_data.append(model_data)
        
        # 合并排序后的数据
        if ordered_data:
            reordered_scores = pd.concat(ordered_data, ignore_index=True)
        else:
            reordered_scores = avg_scores.copy()
        
        # 设置雷达图参数
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # 绘制每个模型的数据
        for _, (_, row) in enumerate(reordered_scores.iterrows()):
            model_name = row['model']
            values = row[metrics].tolist()
            values += values[:1]
            
            # 获取颜色，如果未指定则使用默认颜色
            color = color_map.get(model_name, plt.cm.tab10(_))
            
            # 设置线条属性
            if model_name in ['Graphia-seq', 'Graphia']:
                linewidth = 4
                alpha = 1.0
                zorder = 10  # 确保在最上层
                markersize = 10
            else:
                linewidth = 2.5
                alpha = 0.85
                zorder = 5
                markersize = 8
            
            ax.plot(angles, values, 'o-', linewidth=linewidth, 
                    label=model_name, color=color, markersize=markersize,
                    alpha=alpha, zorder=zorder, markeredgecolor='white', markeredgewidth=1.5)
            ax.fill(angles, values, alpha=0.15, color=color, zorder=zorder-1)
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_metrics, fontsize=26, fontweight='bold', zorder=20)
        
        # 设置图表样式
        ax.set_ylim(0, 1)
        # 修改径向标签（0.2, 0.4等）的大小
        ax.tick_params(axis='y', labelsize=8)  # 设置径向标签字体大小
        ax.grid(True, alpha=0.3)
        
        # 格式化数据集名称（下划线变空格，首字母大写）
        formatted_dataset_name = dataset.replace('_', ' ').title()
        # 设置标题
        ax.set_title(f'{formatted_dataset_name}', size=12, fontweight='bold', pad=20)
        
        # 美化网格
        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('#f8f9fa')
    
    # 隐藏多余的子图
    for idx in range(n_datasets, len(axes)):
        fig.delaxes(axes[idx])
    
    # 如果有空余的子图位置，将第一个空余位置用于放置图例
    # if True:
    #     # 清空第一个空余子图并用于放置图例
    #     empty_ax = axes[n_datasets]
    #     empty_ax.clear()
    #     empty_ax.set_xticks([])
    #     empty_ax.set_yticks([])
    #     empty_ax.spines['polar'].set_visible(False)
    #     empty_ax.set_facecolor('white')
        
    #     # 获取图例句柄和标签
    #     handles, labels = axes[0].get_legend_handles_labels()
    #     # 在空余位置添加图例
    #     empty_ax.legend(handles, labels, loc='center', fontsize=11, 
    #                    frameon=True, fancybox=True, shadow=True, ncol=1)
    #     plt.legend(loc='lower right',  bbox_to_anchor=(0.8, 0.02), 
    #                fontsize=16, frameon=True, fancybox=True, shadow=True, ncol=1)
    #     empty_ax.set_title('Legend', size=12, fontweight='bold')
    # else:
    #     # 如果没有空余位置，在最后一个子图中添加图例（但需要小心遮挡）
    #     # 获取图例句柄和标签
    #     handles, labels = axes[-1].get_legend_handles_labels()
    # 在最后一个子图的空白区域添加图例
    # 添加统一图例（放在图表下方）
    # 获取第一个子图的句柄和标签
    handles, labels = axes[0].get_legend_handles_labels()
    # 创建统一图例，放在图表下方
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.015), ncols = 2,
               fontsize=16, frameon=True, fancybox=True, shadow=True, ncol=3)
    
    # 设置整体标题
    # fig.suptitle('TDGG Social Fidelity Scores Across All Datasets', 
    #              fontsize=16, fontweight='bold', y=0.95)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(output_dir) / "formatted_tdgg_combined_radar.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 已保存所有数据集的组合格式化雷达图至: {output_path}")

# 更新main函数
def main():
    parser = argparse.ArgumentParser(description="绘制 TDGG Social Fidelity Scores 雷达图")
    parser.add_argument("--scores_file", type=str,
                        default="Graphia/reports/tdgg_social_fidelity_scores.csv",
                        help="评分文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="Graphia/reports/figures/",
                        help="图片输出目录")
    
    args = parser.parse_args()
    
    try:
        # 绘制所有数据集的组合格式化雷达图
        plot_formatted_dataset_radar_combined(args.scores_file, args.output_dir)
        
        # 绘制格式化整体雷达图
        plot_formatted_overall_radar(args.scores_file, args.output_dir)
        
        # 绘制评分分布图
        plot_score_distributions(args.scores_file, args.output_dir)
        
        print("🎉 所有图表绘制完成！")
        
    except Exception as e:
        print(f"❌ 绘制图表时出错: {e}")
# 更新main函数
def main():
    parser = argparse.ArgumentParser(description="绘制 TDGG Social Fidelity Scores 雷达图")
    parser.add_argument("--scores_file", type=str,
                        default="Graphia/reports/tdgg_social_fidelity_scores.csv",
                        help="评分文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="Graphia/reports/figures/",
                        help="图片输出目录")
    
    args = parser.parse_args()
    
    try:
        # 绘制所有数据集的组合格式化雷达图
        plot_formatted_dataset_radar_combined(args.scores_file, args.output_dir)
        
        # 绘制格式化整体雷达图
        plot_formatted_overall_radar(args.scores_file, args.output_dir)
        
        # 绘制评分分布图
        plot_score_distributions(args.scores_file, args.output_dir)
        
        print("🎉 所有图表绘制完成！")
        
    except Exception as e:
        print(f"❌ 绘制图表时出错: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# 读取CSV文件
df = pd.read_csv('Graphia/reports/concat/merged_edge_matrix_cut.csv')


# 对 edge_df 应用重命名规则
def rename_edge_model(model_name):
    if re.match(r'grpo_.*_sotopia_edge_.*', model_name):
        return 'Graphia-seq'
    elif model_name.startswith('grpo_'):
        return 'Graphia'
    return model_name

# 筛选需要的模型和列
model_rename_map = {
    'qwen3_sft': 'Qwen3-8B-SFT',
    'qwen3': 'Qwen3-8B',
    'DGGen': 'DGGen',
    'DYMOND': 'DYMOND',
    'tigger': 'Tigger',
    'idgg_csv_processed': 'GAG-general'
}
# 重命名数据集
dataset_rename_map = {
    '8days_dytag_small_text_en': 'Propagate-En',
   
}
df['dataset'] = df['dataset'].replace(dataset_rename_map)
df['model'] = df['model'].replace(model_rename_map)
df['model'] = df['model'].apply(rename_edge_model)
df['average'] *= 5
models_to_plot = ['Qwen3-8B', 'Qwen3-8B-SFT','Graphia-seq', 'Graphia']
dimension_metrics = ['GF', 'CF', 'PD', 'DA', 'IQ', 'CR']  # 不包括average
all_metrics = dimension_metrics + ['average']

# 获取所有唯一的数据集
datasets = df['dataset'].unique()

# 定义改进的颜色映射（使用您提供的配色）
colors = {
    # 'Qwen3-8B': '#EF767A',      # 您提供的红色
    # 'Qwen3-8B-SFT': '#456990',  # 您提供的蓝色
    # 'Graphia-seq': 
    # 'Graphia': '#48C0AA'        # 您提供的绿色
    'Qwen3-8B': '#F2C742',      # 您提供的红色
    'Qwen3-8B-SFT': '#7498B5',  # 您提供的蓝色
    'Graphia-seq': '#E6E6E6',
    'Graphia': '#D66969'        # 您提供的绿色
}

# 创建子图布局
n_datasets = len(datasets)
cols = min(1, n_datasets)
rows = (n_datasets + cols - 1) // cols

# 创建两个图表：详细指标图和汇总对比图
# 1. 详细指标图
fig1, axes1 = plt.subplots(rows, cols, figsize=(10, 6.5*rows))
fig1.patch.set_facecolor('white')

# 处理单个子图的情况
if n_datasets == 1:
    axes1 = [axes1]
elif rows == 1 or cols == 1:
    axes1 = axes1.flatten()
else:
    axes1 = axes1.flatten()

# 为每个数据集绘制详细指标柱状图（不显示数值标签）
for idx, dataset in enumerate(datasets):
    ax = axes1[idx]
    
    # 筛选当前数据集的数据
    dataset_df = df[df['dataset'] == dataset].copy()
    
    # 筛选需要的模型
    filtered_df = dataset_df[dataset_df['model'].isin(models_to_plot)]
    
    # 按模型分组并计算平均值
    avg_scores = filtered_df.groupby('model')[all_metrics].mean()
    
    # 绘制分组柱状图
    x = np.arange(len(all_metrics))
    # 根据模型数量动态调整宽度
    width = 0.8 / len(models_to_plot)  # 总宽度0.8，平均分配给各模型
    
    # 为每个模型绘制柱状图（不添加数值标签）
    for i, (model_name, row) in enumerate(avg_scores.iterrows()):
        offset = (i - len(models_to_plot)/2 + 0.5) * width
        color = colors.get(model_name, '#7f7f7f')  # 默认灰色
        ax.bar(x + offset, row.values, width, label=model_name, 
               color=color, alpha=0.9, edgecolor='white', linewidth=1.2)

    # 改进的标签和样式
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
    ax.set_title(f'{dataset.replace("_", " ").title()}', size=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # 简化边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', which='major', labelsize=11)

# 隐藏多余的子图
for idx in range(n_datasets, len(axes1)):
    fig1.delaxes(axes1[idx])

# 改进的图例（增大字体）
handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08), 
           fontsize=16, frameon=True, fancybox=True, shadow=True, ncol=2,
           columnspacing=3, handletextpad=1, markerscale=1.5)

# 更合理的布局间距
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 保存详细指标图
output_path1 = 'Graphia/reports/figures/model_comparison_detailed_metrics.pdf'
plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

# 2. 汇总对比图（包含average总分和6个维度的对比）
# 计算每个模型在所有数据集上的平均分
overall_avg_scores = df[df['model'].isin(models_to_plot)].groupby('model')[all_metrics].mean()

# 创建汇总对比图（两个子图并排）
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
fig2.patch.set_facecolor('white')

all_width = 1.2

# 左侧子图：Average总分
avg_scores_only = overall_avg_scores['average'].reindex(models_to_plot)
# 根据模型数量动态调整宽度
width1 = all_width / len(models_to_plot) if len(models_to_plot) > 1 else all_width
x1 = np.arange(len(models_to_plot))
bars1 = ax1.bar(x1, avg_scores_only.values, 
                width=width1,
                color=[colors.get(model, '#7f7f7f') for model in models_to_plot],
                alpha=0.9, edgecolor='white', linewidth=1.5)

# 添加数值标签
for bar, value in zip(bars1, avg_scores_only.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.08,
             f'{value:.2f}', ha='center', va='bottom', fontsize=24, fontweight='bold')

# 设置左侧子图标签和样式
# ax1.set_xlabel('Models', fontsize=16, fontweight='bold')
ax1.set_ylabel('Average Score', fontsize=20, fontweight='bold')
ax1.set_title('Models', size=24, fontweight='bold')
ax1.set_ylim(0, 5.5)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_axisbelow(True)
ax1.set_xticks(x1)
ax1.set_xticklabels(models_to_plot, fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', which='major', labelsize=24)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#cccccc')
ax1.spines['bottom'].set_color('#cccccc')

# 右侧子图：6个维度对比（显示数值标签）
x2 = np.arange(len(dimension_metrics))
# 根据模型数量动态调整宽度
width2 = 0.8 / len(models_to_plot)  # 总宽度0.8，平均分配给各模型

# 为每个模型绘制柱状图并添加数值标签
for i, model_name in enumerate(models_to_plot):
    offset = (i - len(models_to_plot)/2 + 0.5) * width2
    color = colors.get(model_name, '#7f7f7f')
    values = overall_avg_scores.loc[model_name, dimension_metrics].values
    bars2 = ax2.bar(x2 + offset, values, width2, label=model_name,
                    color=color, alpha=0.9, edgecolor='white', linewidth=1.2)
    
    # 添加数值标签
    for bar, value in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# # 设置右侧子图标签和样式
# ax2.set_xlabel('Dimensions', fontsize=16, fontweight='bold')
ax2.set_ylabel('Scores', fontsize=20, fontweight='bold')
ax2.set_title('Dimensions', size=24, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(dimension_metrics, fontsize=24, fontweight='bold')
ax2.set_ylim(0, 5.5)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cccccc')
ax2.spines['bottom'].set_color('#cccccc')
ax2.tick_params(axis='y', which='major', labelsize=24)

# 添加图例
ax2.legend(fontsize=22, frameon=True, fancybox=True, shadow=True, ncol=4,
           columnspacing=2, handletextpad=1)

# 调整子图间距
plt.subplots_adjust(wspace=0.1)

# 保存汇总对比图
output_path2 = 'Graphia/reports/figures/model_overall_comparison.pdf'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"✅ 已保存详细指标图至: {output_path1}")
print(f"✅ 已保存汇总对比图至: {output_path2}")
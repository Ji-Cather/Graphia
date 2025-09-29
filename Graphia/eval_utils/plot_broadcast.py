import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from Graphia.utils.bwr_ctdg import BWRCTDGALLDataset, Dataset_Template


def create_broadcast_comparison_plot(save_path='Graphia/reports/figures'):
    """
    创建broadcast前后的message category传播对比图
    
    Args:
        save_path (str): 保存图表的路径
    """
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 读取四组数据
    # grpo_weibo_daily数据集
    daily_broadcast = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen_broadcast.csv')
    daily_normal = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen.csv')
    
    # grpo_weibo_tech数据集
    tech_broadcast = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen_broadcast.csv')
    tech_normal = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen.csv')
    
    # 假设数据中有'edge_label'列表示message category
    # 如果列名不同，请相应调整
    
    # 统计各数据集中的category分布
    def get_category_distribution(df, dataset_name, method):
        if 'edge_label' in df.columns:
            category_counts = df['edge_label'].value_counts()

        else:
            # 如果没有明确的类别列，假设第一列是类别信息
            category_counts = df.iloc[:, 0].value_counts()
            
        df_result = category_counts.reset_index()
        df_result.columns = ['category', 'count']
        df_result['dataset'] = dataset_name
        df_result['method'] = method
        return df_result
    
    # 获取各类别分布
    daily_broadcast_dist = get_category_distribution(daily_broadcast, 'Daily', 'With Broadcast')
    daily_normal_dist = get_category_distribution(daily_normal, 'Daily', 'Without Broadcast')
    tech_broadcast_dist = get_category_distribution(tech_broadcast, 'Tech', 'With Broadcast')
    tech_normal_dist = get_category_distribution(tech_normal, 'Tech', 'Without Broadcast')
    
    # 合并所有数据
    all_data = pd.concat([daily_broadcast_dist, daily_normal_dist, 
                         tech_broadcast_dist, tech_normal_dist], ignore_index=True)
    
    # 创建对比图（只显示差异图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Message Category Distribution: Broadcast Effect (Difference)', fontsize=20)
    
    # 设置更大的字体
    plt.rcParams.update({'font.size': 14})
    
    # Daily数据集差异
    daily_data = all_data[all_data['dataset'] == 'Daily']
    daily_pivot = daily_data.pivot(index='category', columns='method', values='count').fillna(0)
    
    if 'With Broadcast' in daily_pivot.columns and 'Without Broadcast' in daily_pivot.columns:
        daily_diff = daily_pivot['With Broadcast'] - daily_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = ['red' if x < 0 else 'green' for x in daily_diff]
        bars1 = axes[0].bar(range(len(daily_diff)), daily_diff.values, color=colors, alpha=0.7)
        axes[0].set_title('Daily Dataset: Broadcast Effect', fontsize=18)
        axes[0].set_xlabel('Message Category', fontsize=16)
        axes[0].set_ylabel('Count Difference', fontsize=16)
        axes[0].set_xticks(range(len(daily_diff)))
        axes[0].set_xticklabels(daily_diff.index, rotation=45, ha='right', fontsize=12)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, daily_diff.values)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value >= 0 else -1), 
                        f'{int(value)}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=16)
    
    # Tech数据集差异
    tech_data = all_data[all_data['dataset'] == 'Tech']
    tech_pivot = tech_data.pivot(index='category', columns='method', values='count').fillna(0)
    
    if 'With Broadcast' in tech_pivot.columns and 'Without Broadcast' in tech_pivot.columns:
        tech_diff = tech_pivot['With Broadcast'] - tech_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = ['red' if x < 0 else 'green' for x in tech_diff]
        bars2 = axes[1].bar(range(len(tech_diff)), tech_diff.values, color=colors, alpha=0.7)
        axes[1].set_title('Tech Dataset: Broadcast Effect', fontsize=18)
        axes[1].set_xlabel('Message Category', fontsize=16)
        axes[1].set_ylabel('Count Difference', fontsize=16)
        axes[1].set_xticks(range(len(tech_diff)))
        axes[1].set_xticklabels(tech_diff.index, rotation=45, ha='right', fontsize=12)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars2, tech_diff.values)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value >= 0 else -1), 
                        f'{int(value)}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=16)
    
    plt.tight_layout()
    
    # 保存图表
    save_file = os.path.join(save_path, 'broadcast_difference_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_file}")
    
    plt.show()
    
    # 输出统计摘要
    print("=== 数据集统计摘要 ===")
    print(f"Daily Dataset - With Broadcast: {len(daily_broadcast)} 条边")
    print(f"Daily Dataset - Without Broadcast: {len(daily_normal)} 条边")
    print(f"Tech Dataset - With Broadcast: {len(tech_broadcast)} 条边")
    print(f"Tech Dataset - Without Broadcast: {len(tech_normal)} 条边")
    
    return all_data

def create_normalized_comparison(save_path='Graphia/reports/figures'):
    """
    创建标准化的对比图（百分比形式）
    
    Args:
        save_path (str): 保存图表的路径
    """
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 读取数据
    daily_broadcast = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen_broadcast.csv')
    daily_normal = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen.csv')
    tech_broadcast = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen_broadcast.csv')
    tech_normal = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen.csv')
    
    # 计算百分比分布
    def get_percentage_distribution(df, dataset_name, method):
        if 'edge_label' in df.columns:
            category_counts = df['edge_label'].value_counts()

        else:
            category_counts = df.iloc[:, 0].value_counts()
            
        total = category_counts.sum()
        category_pct = category_counts / total * 100
        
        df_result = category_pct.reset_index()
        df_result.columns = ['category', 'percentage']
        df_result['dataset'] = dataset_name
        df_result['method'] = method
        return df_result
    
    # 获取百分比分布
    daily_broadcast_pct = get_percentage_distribution(daily_broadcast, 'Daily', 'With Broadcast')
    daily_normal_pct = get_percentage_distribution(daily_normal, 'Daily', 'Without Broadcast')
    tech_broadcast_pct = get_percentage_distribution(tech_broadcast, 'Tech', 'With Broadcast')
    tech_normal_pct = get_percentage_distribution(tech_normal, 'Tech', 'Without Broadcast')
    
    # 合并数据
    all_pct_data = pd.concat([daily_broadcast_pct, daily_normal_pct, 
                              tech_broadcast_pct, tech_normal_pct], ignore_index=True)
    
    # 创建标准化对比图（只显示差异图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Message Category Distribution (%): Broadcast Effect (Difference)', fontsize=20)
    
    # 设置更大的字体
    plt.rcParams.update({'font.size': 14})
    
    # Daily数据集差异
    daily_pct = all_pct_data[all_pct_data['dataset'] == 'Daily']
    daily_pct_pivot = daily_pct.pivot(index='category', columns='method', values='percentage').fillna(0)
    
    if 'With Broadcast' in daily_pct_pivot.columns and 'Without Broadcast' in daily_pct_pivot.columns:
        daily_pct_diff = daily_pct_pivot['With Broadcast'] - daily_pct_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = ['red' if x < 0 else 'green' for x in daily_pct_diff]
        bars1 = axes[0].bar(range(len(daily_pct_diff)), daily_pct_diff.values, color=colors, alpha=0.7)
        axes[0].set_title('Daily Dataset: Broadcast Effect (%)', fontsize=18)
        axes[0].set_xlabel('Message Category', fontsize=16)
        axes[0].set_ylabel('Percentage Difference (%)', fontsize=16)
        axes[0].set_xticks(range(len(daily_pct_diff)))
        axes[0].set_xticklabels(daily_pct_diff.index, rotation=45, ha='right', fontsize=12)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, daily_pct_diff.values)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if value >= 0 else -0.1), 
                        f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=16)
    
    # Tech数据集差异
    tech_pct = all_pct_data[all_pct_data['dataset'] == 'Tech']
    tech_pct_pivot = tech_pct.pivot(index='category', columns='method', values='percentage').fillna(0)
    
    if 'With Broadcast' in tech_pct_pivot.columns and 'Without Broadcast' in tech_pct_pivot.columns:
        tech_pct_diff = tech_pct_pivot['With Broadcast'] - tech_pct_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = ['red' if x < 0 else 'green' for x in tech_pct_diff]
        bars2 = axes[1].bar(range(len(tech_pct_diff)), tech_pct_diff.values, color=colors, alpha=0.7)
        axes[1].set_title('Tech Dataset: Broadcast Effect (%)', fontsize=18)
        axes[1].set_xlabel('Message Category', fontsize=16)
        axes[1].set_ylabel('Percentage Difference (%)', fontsize=16)
        axes[1].set_xticks(range(len(tech_pct_diff)))
        axes[1].set_xticklabels(tech_pct_diff.index, rotation=45, ha='right', fontsize=12)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars2, tech_pct_diff.values)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if value >= 0 else -0.1), 
                        f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=16)
    
    plt.tight_layout()
    
    # 保存图表
    save_file = os.path.join(save_path, 'broadcast_difference_percentage.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_file}")
    
    plt.show()

def create_broadcast_comparison_with_labels(save_path='Graphia/reports/figures'):
    """
    创建broadcast前后的message category传播对比图，使用实际的标签文本
    
    Args:
        save_path (str): 保存图表的路径
    """
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 读取四组数据
    daily_broadcast = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen_broadcast.csv')
    daily_normal = pd.read_csv('Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen.csv')
    tech_broadcast = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen_broadcast.csv')
    tech_normal = pd.read_csv('Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen.csv')
    
    tech_data = BWRCTDGALLDataset(
        root="Graphia/data/weibo_tech"
    )
    daily_data = BWRCTDGALLDataset(
        root="Graphia/data/weibo_daily"
    )
    type_color_map = {
        "comment": "#F79927",
        "repost": "#2F5763"
    }
    # 创建映射函数，将edge_label的数值索引转换为文本
    def map_label_to_text(df, label_text_list, column_name='edge_label'):
        # 创建一个映射字典，索引对应文本
        label_mapping = {i: text for i, text in enumerate(label_text_list)}
        # 应用映射
        df_text = df.copy()
        df_text[column_name] = df[column_name].map(label_mapping)
        return df_text
    
    # 将edge_label的数值转换为文本
    tech_normal = map_label_to_text(tech_normal, tech_data.data.label_text)
    tech_broadcast = map_label_to_text(tech_broadcast, tech_data.data.label_text)
    tech_broadcast_info = Dataset_Template["weibo_tech"]["broadcast_message"]
    daily_normal = map_label_to_text(daily_normal, daily_data.data.label_text)
    daily_broadcast = map_label_to_text(daily_broadcast, daily_data.data.label_text)
    daily_broadcast_info = Dataset_Template["weibo_daily"]["broadcast_message"]
    
    # 统计各数据集中的category分布
    def get_category_distribution_with_text(df, dataset_name, method):
        if 'edge_label' in df.columns:
            category_counts = df['edge_label'].value_counts()
        else:
            # 如果没有明确的类别列，假设第一列是类别信息
            category_counts = df.iloc[:, 0].value_counts()
            
        df_result = category_counts.reset_index()
        df_result.columns = ['category', 'count']
        df_result['dataset'] = dataset_name
        df_result['method'] = method
        return df_result
    
    # 获取各类别分布
    daily_broadcast_dist = get_category_distribution_with_text(daily_broadcast, 'Daily', 'With Broadcast')
    daily_normal_dist = get_category_distribution_with_text(daily_normal, 'Daily', 'Without Broadcast')
    tech_broadcast_dist = get_category_distribution_with_text(tech_broadcast, 'Tech', 'With Broadcast')
    tech_normal_dist = get_category_distribution_with_text(tech_normal, 'Tech', 'Without Broadcast')
    
    # 合并所有数据
    all_data = pd.concat([daily_broadcast_dist, daily_normal_dist, 
                         tech_broadcast_dist, tech_normal_dist], ignore_index=True)
    
    # 创建对比图（只显示差异图）
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # 调整高度从14到12
    # fig.suptitle('Conditional Message Impact on Social Network Propagation', fontsize=24, weight='bold', y=0.95)
    
    # 设置更大的字体
    plt.rcParams.update({'font.size': 16})
    
    # Daily数据集差异
    daily_data_filtered = all_data[all_data['dataset'] == 'Daily']
    daily_pivot = daily_data_filtered.pivot(index='category', columns='method', values='count').fillna(0)
    
    if 'With Broadcast' in daily_pivot.columns and 'Without Broadcast' in daily_pivot.columns:
        daily_diff = daily_pivot['With Broadcast'] - daily_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = [type_color_map.get(cat, '#CCCCCC') for cat in daily_diff.index]
        bars1 = axes[0, 0].bar(range(len(daily_diff)), daily_diff.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('Weibo Daily: Boosting Comments', fontsize=22, weight='bold', pad=15)
        axes[0, 0].set_ylabel('Count Difference', fontsize=20, weight='bold')
        axes[0, 0].set_xticks(range(len(daily_diff)))
        axes[0, 0].tick_params(axis='both', which='major', labelsize=18)
        axes[0, 0].set_xticklabels(daily_diff.index, ha='center', fontsize=20, weight='bold')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        axes[0, 0].spines['left'].set_linewidth(1.5)
        axes[0, 0].spines['bottom'].set_linewidth(1.5)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, daily_diff.values)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(daily_diff.values)*0.02 if value >= 0 else -max(daily_diff.values)*0.02), 
                        f'{int(value)}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=18, weight='bold')
    
    # Tech数据集差异
    tech_data_filtered = all_data[all_data['dataset'] == 'Tech']
    tech_pivot = tech_data_filtered.pivot(index='category', columns='method', values='count').fillna(0)
    
    if 'With Broadcast' in tech_pivot.columns and 'Without Broadcast' in tech_pivot.columns:
        tech_diff = tech_pivot['With Broadcast'] - tech_pivot['Without Broadcast']
        # 使用新的配色方案
        colors = [type_color_map.get(cat, '#CCCCCC') for cat in tech_diff.index]
        bars2 = axes[0, 1].bar(range(len(tech_diff)), tech_diff.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('Weibo Tech: Boosting Reposts', fontsize=22, weight='bold', pad=15)
        axes[0, 1].set_ylabel('Count Difference', fontsize=20, weight='bold')
        axes[0, 1].set_xticks(range(len(tech_diff)))
        axes[0, 1].tick_params(axis='both', which='major', labelsize=18)
        axes[0, 1].set_xticklabels(tech_diff.index, ha='center', fontsize=20, weight='bold')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        axes[0, 1].spines['left'].set_linewidth(1.5)
        axes[0, 1].spines['bottom'].set_linewidth(1.5)
        
        # 在柱状图上添加数值标签
        for i, (bar, value) in enumerate(zip(bars2, tech_diff.values)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(tech_diff.values)*0.02 if value >= 0 else -max(tech_diff.values)*0.02), 
                        f'{int(value)}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=18, weight='bold')
    
    # 添加Daily数据集的broadcast信息
    daily_info_text = """Attention Weibo users! 
Special comment-to-win event unlocked!
[Comment] on any post using #CommentChallenge"""
    
    axes[1, 0].text(0.5, 0.88, daily_info_text, 
                    transform=axes[1, 0].transAxes, 
                    fontsize=18, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=type_color_map['comment'], 
                              alpha=0.7, edgecolor='white', linewidth=2),
                    weight='bold')
    axes[1, 0].axis('off')  # 移除子图标题
    
    # 添加Tech数据集的broadcast信息
    tech_info_text = """Dear Weibo Users!
To celebrate the New Year
Weibo's launching an epic campaign!
[Repost] & @3 friends for iPhone 15 Pro Max!"""
    
    axes[1, 1].text(0.5, 0.88, tech_info_text, transform=axes[1, 1].transAxes, 
                    fontsize=18, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=type_color_map['repost'], 
                              alpha=0.7, edgecolor='white', linewidth=2),
                    weight='bold')
    axes[1, 1].axis('off')  # 移除子图标题
    
    plt.tight_layout(pad=2.0)  # 减少padding
    
    # 保存图表
    save_file = os.path.join(save_path, 'broadcast_difference_with_labels.pdf')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_file}")
    
    plt.show()
    
    return all_data

# 使用示例
if __name__ == "__main__":
    try:
        # 创建差异对比图
        # comparison_data = create_broadcast_comparison_plot()
        
        # # 创建标准化百分比差异对比图
        # create_normalized_comparison()
        
        # 创建带标签文本的差异对比图
        create_broadcast_comparison_with_labels()
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确认以下文件路径是否正确:")
        print("1. Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen_broadcast.csv")
        print("2. Graphia/results/grpo_weibo_daily_rl-all-domain_100epoch/weibo_daily/test/inference/edge_ggen.csv")
        print("3. Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen_broadcast.csv")
        print("4. Graphia/results/grpo_weibo_tech_rl-all-domain_100epoch/weibo_tech/test/inference/edge_ggen.csv")
    except Exception as e:
        print(f"绘图过程中出现错误: {e}")
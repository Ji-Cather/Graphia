import pandas as pd
import numpy as np

# Step 1: 手动构建数据（因为原始输入是 LaTeX 表格）
# 每行对应一个 model + dataset 的记录
data = [
    # Propagate-En
    {"model": "DGGen",       "dataset": "Propagate-En", "m_struct_score": 0.6602,   "m_struct_rank": 2.0,      "m_phenom_score": 0.5142,   "m_phenom_rank": 3.0,      "idgg_score": 0.5872,     "idgg_rank": 2.0},
    {"model": "DYMOND",      "dataset": "Propagate-En", "m_struct_score": 0.1147,   "m_struct_rank": 4.75,     "m_phenom_score": 0.2916,   "m_phenom_rank": 4.3333,   "idgg_score": 0.2032,     "idgg_rank": 6.0},
    {"model": "Tigger",      "dataset": "Propagate-En", "m_struct_score": 0.252,    "m_struct_rank": 4.0,      "m_phenom_score": 0.459,    "m_phenom_rank": 3.3333,   "idgg_score": 0.3555,     "idgg_rank": 3.0},
    {"model": "Qwen3-8b-sft","dataset": "Propagate-En", "m_struct_score": 0.4191,   "m_struct_rank": 4.0,      "m_phenom_score": 0.2649,   "m_phenom_rank": 4.3333,   "idgg_score": 0.342,      "idgg_rank": 5.0},
    {"model": "LLMGGen-seq", "dataset": "Propagate-En", "m_struct_score": 0.4371,   "m_struct_rank": 3.5,      "m_phenom_score": 0.25,     "m_phenom_rank": 4.3333,   "idgg_score": 0.3436,     "idgg_rank": 4.0},
    {"model": "LLMGGen",     "dataset": "Propagate-En", "m_struct_score": 0.9376,   "m_struct_rank": 1.75,     "m_phenom_score": 1.0,      "m_phenom_rank": 1.0,      "idgg_score": 0.9688,     "idgg_rank": 1.0},

    # Weibo Daily
    {"model": "DGGen",       "dataset": "Weibo Daily", "m_struct_score": 0.244,     "m_struct_rank": 4.75,     "m_phenom_score": 0.5843,   "m_phenom_rank": 2.6667,   "idgg_score": 0.4141,     "idgg_rank": 5.0},
    {"model": "Tigger",      "dataset": "Weibo Daily", "m_struct_score": 0.353,     "m_struct_rank": 3.75,     "m_phenom_score": 0.3333,   "m_phenom_rank": 3.6667,   "idgg_score": 0.3431,     "idgg_rank": 6.0},
    {"model": "GAG-general", "dataset": "Weibo Daily", "m_struct_score": 0.5034,    "m_struct_rank": 2.75,     "m_phenom_score": 0.4057,   "m_phenom_rank": 4.3333,   "idgg_score": 0.4546,     "idgg_rank": 3.0},
    {"model": "Qwen3-8b-sft","dataset": "Weibo Daily", "m_struct_score": 0.3328,   "m_struct_rank": 4.0,      "m_phenom_score": 0.683,    "m_phenom_rank": 2.3333,   "idgg_score": 0.5079,     "idgg_rank": 2.0},
    {"model": "LLMGGen-seq", "dataset": "Weibo Daily", "m_struct_score": 0.3344,   "m_struct_rank": 4.0,      "m_phenom_score": 0.5455,   "m_phenom_rank": 3.6667,   "idgg_score": 0.4399,     "idgg_rank": 4.0},
    {"model": "LLMGGen",     "dataset": "Weibo Daily", "m_struct_score": 1.0,       "m_struct_rank": 1.0,      "m_phenom_score": 0.7765,   "m_phenom_rank": 2.6667,   "idgg_score": 0.8882,     "idgg_rank": 1.0},

    # Weibo Tech
    {"model": "DGGen",       "dataset": "Weibo Tech", "m_struct_score": 0.1597,    "m_struct_rank": 4.75,     "m_phenom_score": 0.5742,   "m_phenom_rank": 4.0,      "idgg_score": 0.3669,     "idgg_rank": 5.0},
    {"model": "Tigger",      "dataset": "Weibo Tech", "m_struct_score": 0.2172,    "m_struct_rank": 4.5,      "m_phenom_score": 0.2985,   "m_phenom_rank": 5.0,      "idgg_score": 0.2578,     "idgg_rank": 6.0},
    {"model": "GAG-general", "dataset": "Weibo Tech", "m_struct_score": 0.5422,    "m_struct_rank": 2.75,     "m_phenom_score": 0.3294,   "m_phenom_rank": 4.3333,   "idgg_score": 0.4358,     "idgg_rank": 4.0},
    {"model": "Qwen3-8b-sft","dataset": "Weibo Tech", "m_struct_score": 0.1606,   "m_struct_rank": 3.75,     "m_phenom_score": 0.8103,   "m_phenom_rank": 3.0,      "idgg_score": 0.4854,     "idgg_rank": 2.0},
    {"model": "LLMGGen-seq", "dataset": "Weibo Tech", "m_struct_score": 0.1479,   "m_struct_rank": 3.75,     "m_phenom_score": 0.7897,   "m_phenom_rank": 3.0,      "idgg_score": 0.4688,     "idgg_rank": 3.0},
    {"model": "LLMGGen",     "dataset": "Weibo Tech", "m_struct_score": 0.9529,    "m_struct_rank": 1.25,     "m_phenom_score": 1.0,      "m_phenom_rank": 1.0,      "idgg_score": 0.9764,     "idgg_rank": 1.0},
]

# Step 2: 转换为 DataFrame
df = pd.DataFrame(data)

# Step 3: 按 model 分组并计算平均值
grouped = df.groupby('model').agg({
    'm_struct_score': 'mean',
    'm_struct_rank': 'mean',
    'm_phenom_score': 'mean',
    'm_phenom_rank': 'mean',
    'idgg_score': 'mean',
    'idgg_rank': 'mean'
}).round(4)

# Step 4: 输出结果
print("✅ 各模型在三个数据集上的平均指标：\n")
print(grouped)

# Optional: 保存到 CSV 文件
# grouped.to_csv("model_average_scores.csv")

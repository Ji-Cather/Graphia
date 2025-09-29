# Graphia

Graphia 是一个基于强化学习的社会网络图生成框架。

## 目录结构

```
Graphia/
├── scripts/
│   ├── prepare_dataset.sh      # 图数据集格式化脚本
│   ├── train_dp.sh             # 活动预测器训练脚本
│   └── prepare_prompt.sh       # LLM训练数据格式化脚本
├── prompt_data/
│   └── weibo_daily/
│       └── train/
│           ├── cold_start/
│           │   └── combined_examples.jsonl  # SFT训练数据
│           ├── seq/
│           │   ├── seq_edge.jsonl           # Graphia-seq edge rl数据
│           │   └── seq_dst.jsonl            # Graphia-seq dst rl数据
│           └── teacher_forcing/
│               ├── edge_text_examples.jsonl # Graphia edge rl数据
│               └── query_examples.jsonl     # Graphia dst rl数据
└── README.md
```

## 模块介绍

### ROLL 模块

Graphia 依赖于 [ROLL](git@github.com:alibaba/ROLL.git) 进行 LLM 的强化学习训练。我们对原始代码进行了部分修改以适应特定需求。

请将 `rlvr` 部分放置在以下路径：
```
ROLL/roll/pipeline/rlvr
```

## 环境要求

- Python 3.7+
- PyTorch 1.10+
- 相关依赖库（根据实际需求补充）

## 快速开始

### 1. 数据处理流程

通过以下脚本完成数据预处理工作：

```bash
# 格式化图数据集
bash scripts/prepare_dataset.sh

# 训练活动预测器 (Activity Predictor)
bash scripts/train_dp.sh

# 训练奖励模型GNN
bash scripts/train_gnn_tgn.sh

# 格式化LLM训练数据
bash scripts/prepare_prompt.sh
```

各脚本功能说明：
- [prepare_dataset.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/prepare_dataset.sh): 准备和格式化社会网络图数据集
- [train_dp.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/train_dp.sh): 训练用于图节点表示学习的活动预测器
- [prepare_prompt.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/prepare_prompt.sh): 生成用于大语言模型训练的提示数据

### 2. LLM 训练指南

以下以 weibo-tech 数据集为例进行说明，假设已完成上述数据准备步骤。

SFT 训练数据位置：
```
Graphia/prompt_data/weibo_daily/train/cold_start/combined_examples.jsonl
```

#### 强化学习配置

| 训练类型 | 配置文件路径 |
|---------|-------------|
| DST RL | [ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_dst_weibo_tech.yaml](file:///data/jiarui_ji/Graphia/ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_dst_weibo_tech.yaml) |
| Edge RL | [ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_easy_seq_weibo_tech.yaml](file:///data/jiarui_ji/Graphia/ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_easy_seq_weibo_tech.yaml) |

#### 训练执行

相关训练命令请参考：
```bash
ROLL/examples/rlvr_megatron_dst/local_run.sh
```

## 评估脚本

### 图生成后处理

- TDGG 处理：`Graphia/scripts/postprocess_tdgg.sh`
- IDGG 处理：`Graphia/scripts/postprocess_idgg.sh`

### 报告合并与评估

处理完成后，首先合并各个模型的报告，然后进行评估：

```bash
# 合并报告
bash Graphia/scripts/concat_reports.sh

# 执行评估
TDGG评估：Graphia/eval_utils/eval_tdgg.py
IDGG评估：Graphia/eval_utils/eval_idgg.py
```


## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。


## 致谢
感谢以下开源项目和研究团队的支持：
- [ROLL](git@github.com:alibaba/ROLL.git) - 强化学习训练框架
- [GDGB](https://github.com/Lucas-PJ/GDGB-ALGO) - 文本动态图Benchmark
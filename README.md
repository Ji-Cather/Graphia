# Graphia

Graphia is a reinforcement learning-based social network graph generation framework.

## Directory Structure

```
Graphia/
├── scripts/
│   ├── prepare_dataset.sh      # Graph dataset formatting script
│   ├── train_dp.sh             # Activity predictor training script
│   └── prepare_prompt.sh       # LLM training data formatting script
├── prompt_data/
│   └── weibo_daily/
│       └── train/
│           ├── cold_start/
│           │   └── combined_examples.jsonl  # SFT training data
│           ├── seq/
│           │   ├── seq_edge.jsonl           # Graphia-seq edge rl data
│           │   └── seq_dst.jsonl            # Graphia-seq dst rl data
│           └── teacher_forcing/
│               ├── edge_text_examples.jsonl # Graphia edge rl data
│               └── query_examples.jsonl     # Graphia dst rl data
└── README.md
```

## Module Introduction

### ROLL Module

Graphia relies on [ROLL](git@github.com:alibaba/ROLL.git) for LLM reinforcement learning training. We have made some modifications to the original code to meet specific requirements.

Please place the `rlvr` component in the following path:
```
ROLL/roll/pipeline/rlvr
```

## Environment Requirements

- Python 3.7+
- PyTorch 1.10+
- Related dependency libraries (to be supplemented based on actual needs)

## Quick Start

### 1. Data Processing Workflow

Complete data preprocessing through the following scripts:

```bash
# Format graph dataset
bash scripts/prepare_dataset.sh

# Train activity predictor
bash scripts/train_dp.sh

# Train reward model GNN
bash scripts/train_gnn_tgn.sh

# Format LLM training data
bash scripts/prepare_prompt.sh
```

Script function descriptions:
- [prepare_dataset.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/prepare_dataset.sh): Prepare and format social network graph datasets
- [train_dp.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/train_dp.sh): Train activity predictor for graph node representation learning
- [prepare_prompt.sh](file:///data/jiarui_ji/Graphia/Graphia/scripts/prepare_prompt.sh): Generate prompts for large language model training

### 2. LLM Training Guide

The following uses the weibo-tech dataset as an example, assuming the above data preparation steps have been completed.

SFT training data location:
```
Graphia/prompt_data/weibo_daily/train/cold_start/combined_examples.jsonl
```

#### Reinforcement Learning Configuration

| Training Type | Configuration File Path |
|---------------|-------------------------|
| DST RL        | [ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_dst_weibo_tech.yaml](file:///data/jiarui_ji/Graphia/ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_dst_weibo_tech.yaml) |
| Edge RL       | [ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_easy_seq_weibo_tech.yaml](file:///data/jiarui_ji/Graphia/ROLL/examples/rlvr_megatron_dst/rlvr_config_remote_all_easy_seq_weibo_tech.yaml) |

#### Training Execution

Refer to the following for training commands:
```bash
ROLL/examples/rlvr_megatron_dst/local_run.sh
```

## Evaluation Scripts

### Graph Generation Post-processing

- TDGG processing: `Graphia/scripts/postprocess_tdgg.sh`
- IDGG processing: `Graphia/scripts/postprocess_idgg.sh`

### Report Concatenation and Evaluation

After processing, first concatenate reports from different models, then perform evaluation:

```bash
# Concatenate reports
bash Graphia/scripts/concat_reports.sh

# Execute evaluation
TDGG evaluation: Graphia/eval_utils/eval_tdgg.py
IDGG evaluation: Graphia/eval_utils/eval_idgg.py
```

## Contribution Guidelines

Welcome to submit Issues and Pull Requests to help improve the project.

## Acknowledgements

Thanks to the following open-source projects and research teams for their support:
- [ROLL](https://github.com/alibaba/ROLL.git) - Reinforcement learning training framework
- [GDGB](https://github.com/Lucas-PJ/GDGB-ALGO) - Text dynamic graph benchmark

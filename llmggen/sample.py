import pandas as pd

# 采样参数
sample_size = 300
domain_probs = {'dst_rule': 0.6, 'edge_rule': 0.2, 'edge_text_rule': 0.2}

# 按照 domain 分组采样
samples = []
for domain, prob in domain_probs.items():
    n = int(sample_size * prob)
    df_domain = combined_df[combined_df['domain'] == domain]
    samples.append(df_domain.sample(n=n, random_state=0))

# 合并采样结果
interleaved_df = pd.concat(samples, ignore_index=True)
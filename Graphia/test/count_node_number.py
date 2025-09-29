import os
import pandas as pd
import glob

# 获取DyLink_Datasets下所有文件夹
dataset_dirs = [d for d in glob.glob('DyLink_Datasets/*') if os.path.isdir(d)]

results = {}

for dataset_dir in dataset_dirs:
    # 查找edge_list.csv文件
    edge_list_path = os.path.join(dataset_dir, 'edge_list.csv')
    
    if os.path.exists(edge_list_path):
        try:
            # 读取edge_list.csv文件
            df = pd.read_csv(edge_list_path)
            
            # 分别计算u和v的唯一节点数量
            if 'u' in df.columns and 'i' in df.columns:
                # 计算u列的唯一值数量
                unique_u_count = df['u'].nunique()
                
                # 计算v列的唯一值数量
                unique_v_count = df['i'].nunique()
                
                # 存储结果
                data_name = os.path.basename(dataset_dir)
                results[data_name] = {
                    'unique_u_count': unique_u_count,
                    'unique_v_count': unique_v_count
                }
            else:
                print(f"警告: {edge_list_path} 中没有找到'u'和'i'列")
        except Exception as e:
            print(f"处理 {edge_list_path} 时出错: {str(e)}")
    else:
        print(f"警告: 在 {dataset_dir} 中没有找到edge_list.csv文件")

# 打印结果
print("\n节点数量统计结果:")
for dataset, counts in results.items():
    print(f"{dataset}: u列有 {counts['unique_u_count']} 个唯一节点, v列有 {counts['unique_v_count']} 个唯一节点")


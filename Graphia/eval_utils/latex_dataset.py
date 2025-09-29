import pandas as pd

def format_dataset_info_to_latex():
    # 读取CSV文件
    df = pd.read_csv('Graphia/reports/concat/dataset_info.csv')
    
    # 数据集重命名映射
    dataset_rename_map = {
        '8days_dytag_small_text_en': 'Propagate-En',
       
    }
    
    # 应用重命名
    df['dataset_name'] = df['dataset_name'].replace(dataset_rename_map)
    
    # 格式化数据集名称
    def format_dataset_name(name):
        return name.replace('_', ' ').title()
    
    df['dataset_name'] = df['dataset_name'].apply(format_dataset_name)
    
    # 选择需要的列
    columns_to_show = [
        'dataset_name', 'node_count', 'edge_count', 'input_length_day',
        'pred_length_day', 'graph_length_day', 'avg_edge_length_day',
        'cat_num', 'input_edge_length_test', 'prediction_edge_length_test'
    ]
    
    # 创建新的DataFrame只包含需要的列
    df_selected = df[columns_to_show]
    
    # 转置DataFrame
    df_transposed = df_selected.set_index('dataset_name').T
    
    # 格式化索引名称（列名）
    df_transposed.index = df_transposed.index.str.replace('_', ' ').str.title()
    
    # 格式化数值列（添加千位分隔符）
    for col in df_transposed.columns:
        df_transposed[col] = df_transposed[col].apply(
            lambda x: f'{x:,.0f}' if isinstance(x, (int, float)) and x == int(x) 
            else f'{x:,.2f}' if isinstance(x, (int, float)) 
            else str(x)
        )
    
    # 生成LaTeX表格
    latex_table = df_transposed.to_latex(
        caption='Dataset Information',
        label='tab:dataset_info',
        escape=False,
        column_format='l' + 'r' * len(df_transposed.columns)
    )
    
    # 打印LaTeX代码
    print(latex_table)
    
    # 保存到文件
    with open('Graphia/reports/concat/dataset_info.tex', 'w') as f:
        f.write(latex_table)
    
    print("✅ Dataset info LaTeX table saved to: Graphia/reports/concat/dataset_info.tex")

# 执行函数
format_dataset_info_to_latex()
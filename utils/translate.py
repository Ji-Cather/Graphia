import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

def load_translator():
    # 加载预训练的中英翻译模型
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    try:
        # 对文本进行编码
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # 生成翻译
        translated = model.generate(**inputs)
        
        # 解码翻译结果
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"翻译 {text} 时出错: {str(e)}")
        return text

def translate_edge_features(df_path, cols_to_translate):
    # 加载翻译模型
    model, tokenizer = load_translator()
    
    # 读取CSV文件
    df = pd.read_csv(df_path)
    
    # 需要翻译的列
    
    
    # 对每一列进行翻译
    for col in cols_to_translate:
        # 获取唯一值并翻译
        unique_values = df[col].unique()
        translation_dict = {}
        
        for value in tqdm(unique_values, desc=f"翻译 {col}"):
            translation_dict[value] = translate_text(value, model, tokenizer)
        
        # 使用翻译字典替换原值
        df[col] = df[col].map(translation_dict)
    
    # 保存回原文件
    df.to_csv(df_path, index=False)
    print("翻译完成并保存")

if __name__ == "__main__":
    df_path = '/data/jiarui_ji/DTGB/data/8days_dytag_small_text_en/raw/node_feature.csv'
    cols_to_translate = ['portrait_name']
    

    translate_edge_features(df_path, cols_to_translate)

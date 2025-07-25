import gzip
import json 
import numpy as np
import os
import pandas as pd
import torch

from .utils.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from typing import Callable
from typing import List
from typing import Optional


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)



## 不区分seller
class Market(InMemoryDataset, PreprocessingMixin):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        split=None # "head", "n_head", "with_remark"
    ) -> None:
        self.split = split
        
        super(Market, self).__init__(
            root, transform, pre_transform, force_reload
        )
        
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return ["item_description_agg.csv"]
    
    @property
    def processed_file_names(self) -> str:
        return f'{self.split}.pt'
    
    
    def process(self) -> None:
        data = HeteroData()
        item_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]))
        item_data["id"] = np.arange(item_data.shape[0])
        
        # Compute item features
        # 对数值型列使用线性插值填充缺失值
        numeric_cols = item_data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            item_data[col] = item_data[col].interpolate(method='linear')
        
        # 对非数值型列使用"Unknown"填充
        item_data = item_data.fillna("Unknown")
        
        
        if self.split == "head":
            sentences = item_data.apply(
                lambda row:
                    "Tone: " +
                    str(row["diaoxing"]) + "; " +
                    # "Seller: " +
                    # str(row["seller_id"]) + "; " +
                    "Brand: " +
                    str(row["cate_level1_id"]) + "; " +
                    "Categories: " +
                    str(row["cate_id"]) + "; " + 
                    "Price: " +
                    str(row["price"]) + "; " +
                    "Price Range: " +
                    str(row["price_range"]) + "; " +
                    "Coupon Rate: " +
                    str(row.get("zk_rate", "")) + "; " +
                    "Sales: " +
                    str(row.get("med_sales_7d", 0)) + "; " +
                    "CVR: " +
                    str(row.get("med_cvr_7d", 0)) + "; ",
                axis=1
            )
        elif self.split == "n_head":
            sentences = item_data.apply(
                lambda row:
                    str(row["diaoxing"]) + "; " +
                    # "Seller: " +
                    # str(row["seller_id"]) + "; " +
                    str(row["cate_level1_id"]) + "; " +
                    str(row["cate_id"]) + "; " + 
                    str(row["price"]) + "; " +  
                    str(row["price_range"]) + "; " +
                    str(row.get("zk_rate", "")) + "; " +
                    str(row.get("med_sales_7d", 0)) + "; " +
                    str(row.get("med_cvr_7d", 0)) + "; ",
                axis=1
            )
        elif self.split == "with_remark":
            sentences = item_data.apply(
                lambda row:
                    "Tone: " +
                    str(row["diaoxing"]) + "; " +
                    # "Seller: " +
                    # str(row["seller_id"]) + "; " +
                    "Brand: " +
                    str(row["cate_level1_id"]) + "; " +
                    "Categories: " +
                    str(row["cate_id"]) + "; " + 
                    "Price: " +
                    str(row["price"]) + "; " +
                    "Price Range: " +
                    str(row["price_range"]) + "; " +
                    "Coupon Rate: " +
                    str(row.get("zk_rate", "")) + "; " +
                    "Sales: " +
                    str(row.get("med_sales_7d", 0)) + "; " +
                    "CVR: " +
                    str(row.get("med_cvr_7d", 0)) + "; " +
                    "Remark: " +
                    str(row.get("remark", "")) + "; ",
                axis=1
            )
        
        
        item_emb = self._encode_text_feature(sentences)
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])
        



if __name__ == "__main__":
   data = Market("/data/jiarui_ji/RQ-VAE-Recommender/dataset/market", split="n_head")
   data
   

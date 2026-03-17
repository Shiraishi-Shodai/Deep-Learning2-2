import sys
from pathlib import Path

# プロジェクトルート (image2caption/) を取得
root = Path(__file__).resolve().parent.parent.parent
# プロジェクトルートとその親 (Deep-Learning2/) をパスに追加
sys.path.extend([str(root), str(root.parent)])

from common import np
import polars as pl

def accuracy(t_predict_ids, t_ids, word_dict):
    """生成したキャプションと正解キャプションを比較
    """
    t_predict_caption = [word_dict.row(by_predicate=(pl.col("id") == id), named=True)["word"] for id in t_predict_ids] 
    t_caption = [word_dict.row(by_predicate=(pl.col("id") == id), named=True)["word"] for id in t_ids] 

    return t_predict_caption == t_caption
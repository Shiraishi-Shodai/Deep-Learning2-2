import sys
from pathlib import Path

# プロジェクトルート (image2caption/) を取得
root = Path(__file__).resolve().parent.parent.parent
# プロジェクトルートとその親 (Deep-Learning2/) をパスに追加
sys.path.extend([str(root), str(root.parent)])

import kagglehub
import polars as pl
from matplotlib import pyplot as plt
import csv
import cv2
from torch.nn import functional as F
import torch


START_TOKEN = '<start>'
LAST_TOKEN = '<end>'
PAD_TOKEN = '<pad>'

original_txt_file_name = 'captions.txt'
original_csv_file_name = 'captions.csv'
dict_file_name = 'word_dict.csv'

df = None

# =================================
# 1-1. データをダウンロード
# =================================
DATA_DIR = kagglehub.dataset_download("adityajn105/flickr8k")
print(DATA_DIR)

# =================================
# 1-2. polarsのdataframeとして読み込み
# =================================
df = pl.read_csv(rf"{DATA_DIR}\{original_txt_file_name}")
print(df.describe())

# =================================
# 2-1. csvファイルのキャプションに<start> <end> <pad>を追加
# =================================
def start_last_token_check(df, START_TOKEN, LAST_TOKEN, PAD_TOKEN):
    """開始トークン、終了トークン、パディングトークンを追加
    """
    df = df.with_columns(
        pl.when(~pl.col("caption").str.starts_with(START_TOKEN))
        .then(pl.lit(START_TOKEN) + " " + pl.col("caption"))
        .otherwise(pl.col("caption"))
        .alias("caption")
    )

    df = df.with_columns(
        pl.when(~pl.col("caption").str.ends_with(LAST_TOKEN))
        .then(pl.col("caption") + " " + pl.lit(LAST_TOKEN))
        .otherwise(pl.col("caption"))
        .alias("caption")
    )

    df = df.with_columns(
        pl.col("caption").str.split(" ").alias("tokens")
    )

    max_len = df.select(pl.col("tokens").list.len().max()).item()

    df = df.with_columns(
        pl.when(pl.col("tokens").list.len() < max_len)
        .then(
            pl.col("tokens").list.concat(
                pl.lit(PAD_TOKEN).repeat_by(
                    max_len - pl.col("tokens").list.len()
                )
            )
        )
        .otherwise(pl.col("tokens"))
        .alias("tokens")
    )

    # ← CSV保存用に文字列へ戻す
    df = df.with_columns(
        pl.col("tokens").list.join(" ").alias("caption")
    ).drop("tokens")

    return df

# =================================
# 2. csvファイルのキャプションに<start> <end> <pad>を追加
# =================================
df = start_last_token_check(df, START_TOKEN, LAST_TOKEN, PAD_TOKEN)
print(df.describe())
df.write_csv(rf"{DATA_DIR}\{original_csv_file_name}")

# =================================
# 3. 単語IDを辞書化
# =================================
def make_dict(word_series, dict_path):
    """単語の辞書を作成する
    """
    word_dict = {"id": [] , "word" : []}

    for caption in word_series:
        words = caption.split()
        for word in words:
            if word not in word_dict["word"]:
                new_id = next(reversed(word_dict["id"])) + 1 if len(word_dict["id"]) > 0 else 0
                word_dict["id"].append(new_id)
                word_dict["word"].append(word)
    
    word_dataframe = pl.DataFrame(word_dict)
    word_dataframe.write_csv(dict_path)

make_dict(df["caption"], rf"{DATA_DIR}\{dict_file_name}")

# =================================
# one-hotベクトル化
# =================================
# word_dataframe = pl.read_csv(dict_path)
# print(word_dataframe)
# last_id = next(reversed(word_dataframe["id"]))
# print(F.one_hot(torch.arange(0, last_id + 1)))
# print(F.one_hot(torch.arange(0, last_id + 1)).shape)

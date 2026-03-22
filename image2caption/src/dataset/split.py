import random
import polars as pl

def split_dataframe_by_image(df, train_ratio, valid_ratio, seed=42):
    assert train_ratio + valid_ratio < 1.0
    
    unique_image_ids = df["image"].unique().to_list()
    random.Random(42).shuffle(unique_image_ids)

    n_images = len(unique_image_ids)
    n_train = int(n_images * train_ratio)
    n_valid = int(n_images * valid_ratio)

    # 画像名のセットを作成
    train_ids = set(unique_image_ids[:n_train])
    valid_ids = set(unique_image_ids[n_train:n_train + n_valid])
    test_ids = set(unique_image_ids[n_train + n_valid:])

    # データの重複を確認
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    
    # 学習・検証・テストに使用する画像をフィルターしdataframeを作成
    train_df = df.filter(pl.col("image").is_in(train_ids))
    validate_df = df.filter(pl.col("image").is_in(valid_ids))
    test_df = df.filter(pl.col("image").is_in(test_ids))
    
    return train_df, validate_df, test_df
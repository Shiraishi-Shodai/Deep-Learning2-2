# Seq2Seq 学習停止問題の修正

## 1. 問題の概要
`train_seq2seq.py` を実行中、1エポック終了時のバリデーション処理（`eval_seq2seq`）内でプログラムが停止（ハング）する。

## 2. 原因調査の結果

### 調査1: sigmoid関数（部分的な原因）
- **箇所**: `common/functions.py` の `sigmoid` 関数
- **内容**: 整数 `1` をCuPy配列で割る際に `numpy.min_scalar_type` が呼ばれ衝突
- **対処**: `1` → `1.0` に変更（効果あり、ただしこれだけでは不十分）

### 調査2: Decoder.generate()（根本原因）✅
- **箇所**: `ch07_re/seq2seq.py` の `Decoder.generate()` メソッド（85行目付近）
- **根本原因**: **CuPy 0次元配列と NumPy 2.0 の型変換ルールが衝突**

#### 何が起きていたか
```
1回目ループ: sample_id = start_id (Python int) → np.array(int) → 正常
    ↓
np.argmax(score.flatten()) → CuPy 0次元配列を返す
    ↓
2回目ループ: np.array(CuPy 0次元配列) → CuPyが内部でnumpy.min_scalar_typeを呼出 → ハング
```

#### 修正内容
```python
# 修正前
sample_id = np.argmax(score.flatten())     # CuPy 0次元配列
sampled.append(int(sample_id))

# 修正後
sample_id = int(np.argmax(score.flatten())) # Python int に即変換
sampled.append(sample_id)
```

## 3. 修正したファイル
- `common/functions.py`: sigmoid関数の整数→浮動小数点修正
- `ch07_re/seq2seq.py`: `Decoder.generate()` の `np.argmax` 結果を `int()` 変換
- `ch07_re/peeky_seq2seq.py`: `PeekyDecoder.generate()` の同様の修正

## 4. 教訓
NumPy 2.0 + CuPy 環境では、**CuPy 0次元配列（スカラー相当）をPythonのネイティブ型と混在させないこと**が重要。`np.argmax()` や `np.sum()` などスカラーを返す関数の結果は、ループ内で再利用する前に `int()` や `float()` で明示変換すべき。

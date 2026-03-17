# train_seq2seq.py での GPU サポートの有効化

Seq2seq モデルのトレーニングで GPU を使用できるように [train_seq2seq.py](file:///c:/Users/siran/ML/Deep-Learning2/ch07_re/train_seq2seq.py) を修正します。これには、テンソル演算に NumPy の代わりに CuPy を使用するように設定を更新することが含まれます。

## ユーザーレビューが必要

> [!NOTE]
> この変更は、ユーザーが `cupy` をインストール済みであり、互換性のある NVIDIA GPU を持っていること（言及済み）を前提としています。このスクリプトを実行する環境では、CUDA ドライバーと CuPy がセットアップされている必要があります。
>
> [GPUとCuPyの動作確認ガイド](GPUとCuPyの動作確認ガイド.md) を参照して、環境をご確認ください。

## 提案される変更

### `ch07_re`

#### [MODIFY] [train_seq2seq.py](file:///c:/Users/siran/ML/Deep-Learning2/ch07_re/train_seq2seq.py)

- `common.view` をインポートし、`common.np` (および `common.optimizer` や `seq2seq` などそれに依存するモジュール) をインポートする前に、スクリプトの冒頭で `common.config.GPU = True` を設定します。
- これにより、[common/np.py](file:///c:/Users/siran/ML/Deep-Learning2/common/np.py) が `cupy` を `np` としてインポートするようになり、モデルとトレーニングの GPU アクセラレーションが有効になります。

```python
import sys
sys.path.append("..")
# [NEW] GPUを有効化
import common.config
common.config.GPU = True

import numpy as np
...
```

## 検証計画

### 自動テスト
- 現在の環境には GPU がないため、ここでの実行はできません。
- `common.config.GPU = True` が後続のモジュールのインポート前に確実に設定されるようにコード構造を確認します。

### 手動検証
- ユーザーにスクリプトを実行してもらいます: `python train_seq2seq.py`
- 成功すれば、コンソールに `GPU Mode (cupy)` と表示されるはずです（[common/np.py](file:///c:/Users/siran/ML/Deep-Learning2/common/np.py) で定義されています）。
- トレーニング速度は CPU 実行時よりも大幅に高速になるはずです。

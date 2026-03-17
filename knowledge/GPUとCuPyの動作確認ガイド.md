# GPUとCuPyの動作確認ガイド

ご自身のPCでGPU（CUDA）とCuPyが正しくセットアップされているかを確認するための手順です。

## 1. NVIDIA ドライバーとCUDAの確認

ターミナル（PowerShellやCommand Prompt）で以下のコマンドを実行してください。

```powershell
nvidia-smi
```

**確認ポイント:**
- 表が表示され、GPU名（例: NVIDIA GeForce RTX 3060など）が表示されていること。
- 右上の `CUDA Version` が表示されていること（例: 11.x や 12.x）。

> [!NOTE]
> 先ほどあなたの環境で `nvidia-smi` を実行したところ、正常に動作し `CUDA Version: 12.8` と表示されました。ドライバー周りは問題なさそうです。

## 2. CuPyのインストール確認

Python環境に `cupy` パッケージがインストールされているか確認します。

```powershell
pip list | findstr cupy
```

もしインストールされていない場合は、CUDAのバージョンに合わせた `cupy` をインストールする必要があります。
（例: CUDA 12.x の場合）
```powershell
pip install cupy-cuda12x
```

## 3. Python上での動作確認（一番確実な方法）

実際にPythonでGPUが認識されているか、以下のワンライナー（1行スクリプト）で確認できます。

```powershell
python -c "import cupy; print('CuPy Version:', cupy.__version__); print('GPU Device:', cupy.cuda.Device(0).compute_capability)"
```

**成功時:**
- バージョン番号とGPUのCompute Capabilityが表示されます。
- エラーが出なければOKです。

**よくあるエラー:**
- `ModuleNotFoundError`: `cupy` がインストールされていません。
- `ImportError: CuPy is not correctly installed`: CUDAドライバーとCuPyのバージョンが合っていない可能性があります。

---
この確認が完了したら、[train_seq2seq.py](file:///c:/Users/siran/ML/Deep-Learning2/ch07_re/train_seq2seq.py) を実行して学習が高速化されるかお試しください。

# 機械学習研究のための環境構築ガイド (WSL2 + Docker + CUDA)

本ガイドは、**Windows 11 + RTX 5060 Laptop** 搭載PCにおいて、機械学習の研究環境を構築するための手順書です。特に「再現性」と「環境の分離」を重視し、Dockerを使用した構成を採用します。

---

## 1. 全体像と各モジュールの役割

まず、構築する環境の全体像を理解しましょう。各コンポーネントがどのように連携しているか、立ち位置を明確にします。

```mermaid
graph TB
    subgraph Host ["Windows 11 (ホストOS)"]
        direction TB
        Hardware[GPU: RTX 5060 / CPU: Core i7]
        WinDriver[NVIDIA Driver v573.x]
        WL[WSL2 (Linux Kernel)]
    end

    subgraph Docker ["Docker Desktop (コンテナ管理)"]
        Engine[Docker Engine]
    end

    subgraph Container ["Dockerコンテナ (学習環境)"]
        direction TB
        GuestOS[OS: Ubuntu 22.04]
        CUDA[CUDA Toolkit 12.x]
        Lib[PyTorch / TensorFlow]
        Code[あなたのコード (Deep-Learning2)]
    end

    Code --> Lib
    Lib --> CUDA
    CUDA -.->|GPUパススルー| WinDriver
    Engine --> WL
    WL -.->|仮想化| Hardware
```

### 各モジュールの役割解説

1.  **Windows 11 (Host OS)**
    -   **役割**: 物理ハードウェア（CPU、メモリ、GPU）の管理者。
    -   **重要**: ここにインストールした **NVIDIA Driver** が、すべてのGPUアクセスの土台になります。ユーザー環境では `v573.01` が既にインストールされており、準備完了です。

2.  **WSL2 (Windows Subsystem for Linux 2)**
    -   **役割**: Windows上でLinuxを動かすための「軽量な仮想マシン」。
    -   **立ち位置**: Dockerを動かすための基盤として機能します。ユーザー環境では `Ver 2.6.3.0` が動作中であり、準備完了です。

3.  **Docker Desktop**
    -   **役割**: 「コンテナ」という隔離された仮想環境を作成・管理するツール。
    -   **立ち位置**: ユーザー環境では `Ver 29.2.0` がインストール済み。これがWSL2と連携してコンテナを動かします。

4.  **Dockerコンテナ (ここが作業場)**
    -   **役割**: **実際の機械学習環境**。この中に「Ubuntu」「CUDA Toolkit」「PyTorch」「Python」などをすべて閉じ込めます。
    -   **メリット**: もし環境が壊れても、コンテナを削除して作り直すだけで済みます。Windows本体や他のプロジェクトに影響を与えません。

---

## 2. 現在の環境確認（完了済み）

以下のコンポーネントはインストール済みであることを確認しました。

-   [x] **WSL2**: バージョン 2.6.3.0
-   [x] **Docker Desktop**: バージョン 29.2.0
-   [x] **NVIDIA Driver**: バージョン 573.01 (RTX 5060 Laptop対応)
    -   *補足*: Windows側の CUDA Toolkit (v12.9) はDocker環境からは直接参照されませんが、ドライバの一部として機能していれば問題ありません。

---

## 3. プロジェクト作成手順 (PyTorch環境)

ここからが実際の作業です。研究プロジェクトごとに以下の手順を行うことで、クリーンな環境を用意できます。

**作業ディレクトリ例**: `C:\Users\siran\ML\Deep-Learning2`

### ステップ 1: 設定ファイルの作成

プロジェクトフォルダの直下に、以下の2つのファイルを作成します。

#### (1) `Dockerfile`
コンテナの中身（OSやライブラリ）を定義する設計図です。RTX 5060 (最新GPU) に対応するため、新しいCUDAバージョンを指定します。

```dockerfile
# ベースイメージ: PyTorch (CUDA 12.1対応版を使用)
# RTX 50シリーズは新しいアーキテクチャのため、CUDA 12系が必須です。
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# 環境変数の設定 (対話モードを無効化など)
ENV DEBIAN_FRONTEND=noninteractive

# 作業ディレクトリの設定
WORKDIR /workspace

# 必要な基本ツールのインストール
# git: コード管理, curl: 通信確認, vim/nano: ファイル編集
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 自動で生成される __pycache__ を書き込まない設定 (オプション)
ENV PYTHONDONTWRITEBYTECODE=1

# Pythonライブラリの追加インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

#### (2) `docker-compose.yml`
コンテナの起動設定（GPU割り当て、フォルダ共有）を記述します。

```yaml
version: '3.8'

services:
  lab:
    build: .
    image: deep-learning-lab:latest  # 作成されるイメージの名前
    container_name: dl-container      # 起動するコンテナの名前
    
    # GPUの有効化 (NVIDIA Container Toolkit機能)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # ディレクトリ共有
    # ホスト(Windows)の現在のフォルダ(.) を コンテナ内の /workspace にマウント
    volumes:
      - .:/workspace
    
    # ポートフォワーディング (重要: セキュリティ対策)
    # 127.0.0.1を指定することで、外部(インターネット)からの接続を遮断します
    ports:
      - "127.0.0.1:8888:8888" # Jupyter Lab用
    
    # コンテナがすぐに終了しないようにする設定
    stdin_open: true
    tty: true

    # 自動起動コマンド (Jupyter Labを起動)
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='mytoken'
```

#### (3) `requirements.txt`
必要なPythonライブラリを記述します。

```text
matplotlib
pandas
scikit-learn
jupyterlab
# その他、研究に必要なライブラリがあれば追記
```

### ステップ 2: コンテナのビルドと起動

PowerShell または VS Code のターミナルで、プロジェクトフォルダ (`Deep-Learning2`) に移動し、以下のコマンドを実行します。

1.  **ビルドと起動** (初回はダウンロードに時間がかかります)
    ```powershell
    docker compose up -d
    ```

2.  **ログの確認** (Jupyter LabのURLなどが表示されます)
    ```powershell
    docker compose logs -f
    ```

### ステップ 3: 動作確認

ブラウザで `http://127.0.0.1:8888` にアクセスし、設定したトークン (`mytoken`) を入力してログインできれば成功です。

さらに、ノートブック上で以下のコードを実行し、GPUが認識されているか確認してください。

```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
```

**出力例**:
> PyTorch Version: 2.2.0+cu121
> CUDA Available: True
> GPU Name: NVIDIA GeForce RTX 5060 Laptop GPU

---

## 4. よくある質問 (FAQ)とトラブルシューティング

### Q1. WSL上で `nvidia-smi` が動きましたが、これでいいですか？
はい、そのままで大丈夫です。WSL上で `nvidia-smi` がGPU情報を表示できていれば、Dockerコンテナ側にも正しくパススルーされます。

### Q2. コンテナ内のファイルはどこに保存されますか？
`docker-compose.yml` の `volumes` 設定により、コンテナ内の `/workspace` は Windows側のプロジェクトフォルダと同期されています。コンテナ内で作成したファイルは、そのままWindowsのエクスプローラーで見ることができます。コンテナを削除してもファイルは消えません。

### Q3. 「CUDA out of memory」エラーが出たら？
RTX 5060 Laptop のVRAM容量を超えるデータを扱おうとしています。バッチサイズ（`batch_size`）を小さくするか、モデルを軽量化する必要があります。

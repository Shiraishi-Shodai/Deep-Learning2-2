# Image to Caption Project

Flickr8kデータセットを使用して、画像からキャプションを生成するAIプロジェクトです。

## フォルダ構成

```text
image2caption/
├── data/               # データ保存先
│   ├── raw/            # ダウンロードしたままのデータ
│   └── processed/      # 前処理済みのデータ
├── src/                # メインソースコード
│   ├── dataset/        # データ読み込み（flickr8k.py など）
│   ├── models/         # モデル定義（Encoder, Decoder など）
│   ├── training/       # 学習ループと損失関数
│   └── utils/          # 評価指標、可視化ツール
├── notebooks/          # 実験・試行錯誤用スクリプト
├── docs/               # 設計書やメモ
└── .gitignore          # Git管理対象外の設定
```

## パスについて

各スクリプトの冒頭でプロジェクトルートを自動的に `sys.path` に追加するようにしています。
これにより、どのディレクトリから実行しても `src.xxx` という形でインポートが可能です。
また、親ディレクトリにある `common` パッケージも参照できるように設定しています。

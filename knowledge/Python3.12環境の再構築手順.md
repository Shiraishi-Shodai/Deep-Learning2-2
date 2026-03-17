# Python 3.12 環境の再構築手順 (uv 使用)

Python 3.14 で発生していた互換性問題を解決するため、安定した **Python 3.12** を使用して環境を再構築する手順をまとめました。

---

## 🛑 ステップ1：既存の環境をクリーンアップ

まず、現在の Python 3.14 ベースの仮想環境を完全に削除します。

1.  ターミナルを（もし開いていれば）閉じます。
2.  プロジェクトのルートフォルダにある物理的なフォルダを削除します。
    ```powershell
    Remove-Item -Recurse -Force .venv
    ```

---

## 🐍 ステップ2：Python 3.12 で環境を新規作成

`uv` を使って、Python 3.12 を明示的に指定して仮想環境を作成します。

1.  **Python 3.12 の指定と作成**
    ```powershell
    uv venv --python 3.12
    ```
    ※ `uv` は必要に応じて自動的に Python 3.12 をダウンロードしてくれます。

2.  **仮想環境の有効化**
    ```powershell
    .\.venv\Scripts\activate
    ```

---

## 📦 ステップ3：ライブラリの一括インストール

`requirements.txt` に記載されているライブラリと、今回の目的である `cupy-cuda12x` をインストールします。

1.  **requirements.txt の反映**
    ```powershell
    uv pip install -r requirements.txt
    ```

2.  **CuPy のインストール**
    ```powershell
    uv pip install cupy-cuda12x
    ```

---

## ✅ ステップ4：動作確認

正しく設定できたか確認します。

1.  **Pythonバージョンの確認**
    ```powershell
    python --version
    ```
    → `Python 3.12.x` と表示されれば成功です。

2.  **CuPy が GPU を認識しているか確認**
    ```powershell
    python -c "import cupy; print(cupy.cuda.Device(0).attributes)"
    ```
    → GPUのスペック情報が表示されれば、準備完了です！

---

## 💡 セキュリティと保守性のポイント

-   **セキュリティ**: `uv` は依存関係を高速かつ正確に管理するため、サプライチェーン攻撃（悪意のあるパッケージの混入）などのリスクを低減する仕組みが備わっています。
-   **再現性**: `.python-version` という名前のファイルを作り、中に `3.12` とだけ書いて保存しておくと、次回から `uv venv` だけで 3.12 が選ばれるようになり、さらに確実です。

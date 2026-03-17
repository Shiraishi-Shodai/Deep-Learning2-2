import numpy as np
from time_layers import TimeLSTM

def verify_reference_passing():
    # 1. パラメータの準備
    N, D, H, T = 2, 3, 4, 5
    Wx = np.random.randn(D, 4*H)
    Wh = np.random.randn(H, 4*H)
    b = np.random.randn(4*H)
    
    # 2. TimeLSTMレイヤの生成
    layer = TimeLSTM(Wx, Wh, b)
    
    # 3. Forward処理 (内部でLSTMレイヤが作成される)
    xs = np.random.randn(N, T, D)
    layer.forward(xs)
    
    # 4. 検証その1：オブジェクトIDの確認
    # TimeLSTMが持つパラメータと、内部のLSTMレイヤが持つパラメータが同一オブジェクトか確認
    
    print("--- 検証結果 ---")
    
    # TimeLSTMのパラメータ (Wx)
    time_lstm_Wx = layer.params[0]
    
    # 最初の時刻のLSTMレイヤのパラメータ (Wx)
    lstm_layer0_Wx = layer.layers[0].params[0]
    
    print(f"TimeLSTMのWxのID: {id(time_lstm_Wx)}")
    print(f"内部LSTM[0]のWxのID: {id(lstm_layer0_Wx)}")
    
    if id(time_lstm_Wx) == id(lstm_layer0_Wx):
        print("✅ IDが一致しました。同じオブジェクトを参照しています。")
    else:
        print("❌ IDが不一致です。別のオブジェクトです。")

    # 5. 検証その2：値の書き換えによる確認
    print("\n--- 値書き換え検証 ---")
    print(f"変更前のTimeLSTM params[0][0,0]: {layer.params[0][0,0]}")
    print(f"変更前のLSTM[0] params[0][0,0]: {layer.layers[0].params[0][0,0]}")
    
    # TimeLSTM側から値を変更してみる
    val = 999.0
    print(f"TimeLSTM側の値を {val} に変更します...")
    layer.params[0][0,0] = val
    
    print(f"変更後のTimeLSTM params[0][0,0]: {layer.params[0][0,0]}")
    print(f"変更後のLSTM[0] params[0][0,0]: {layer.layers[0].params[0][0,0]}")
    
    if layer.layers[0].params[0][0,0] == val:
         print("✅ 内部LSTMの値も変わりました！参照渡しされています。")
    else:
         print("❌ 内部LSTMの値が変わっていません。コピーされています。")

if __name__ == "__main__":
    verify_reference_passing()
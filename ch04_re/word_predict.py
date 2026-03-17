import numpy as np
import faiss
import pickle

"""
コンテキストを列方向に足し合わせて代表単語ベクトルを作り、全単語とコサイン類似度を取る
もっともコサイン類似度が大きかった単語をターゲット単語とする。
W_inとW_outはどちらも単語の意味を学習している。
正解を1に不正解例を0に近づくように学習している。
"""
# 1. 小さな語彙と埋め込み `W_in` を用意
# 2. 文脈（I, playing, football）の埋め込みを平均して文脈ベクトル `h` を作成 (CBOWのembedと同じ)

#    * 例: `h = [0.55, 0.1167, 0.0]`
# 3. `W_in` と `h` をL2正規化 → 「内積 = コサイン類似度」にする
# 4. （本来は FAISS で）`h_norm` と全単語ベクトルの類似度を一括検索

#    * この環境ではFAISSが使えなかったため NumPy で同じ計算を実行
# 5. 文脈に含まれる単語（I, playing, football）を候補から除外
# 6. 残った候補の上位が予測（今回の例では `like` が最上位）
# --- 1) 学習済みをロード（例） ---
# W_in: (V, H)  float32想定
# ここはあなたの学習コード／保存形式に合わせて用意してください


pkl_filename = "ptb_params.pkl"
with open(pkl_filename, "rb") as f:
    params = pickle.load(f)


word_to_id = params["word_to_id"]
id_to_word = params["id_to_word"]
W_in = params["word_vec"]

# --- 2) L2正規化ユーティリティ(行ごとにxをL2ノルムで正規化する) ---
"""
行ごとにL2ノルムで正規化すると、各単語ベクトルがL2ノルムで正規化されることになる。
すると、コサイン類似度を求める際に、単純に内積をとればコサイン類似度を求められる。
"""
def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True, ord=2) # 行ごとにL2ノルムを計算。行方向にL2ノルムを求めた後の次元数を変化させないようにする。(単語数, 1)
    return x / np.clip(n, eps, None) # 行方向にL2ノルムで割るときに、0除算が起きないようにする。 (単語数, 次元数) / (単語数, 1) = (単語数, 次元数)

W_in = W_in.astype(np.float32)
W_in_norm = l2_normalize(W_in, axis=1)  # (V, H)

# --- 3) FAISSインデックス構築（正確検索：内積） ---
H = W_in_norm.shape[1]
index = faiss.IndexFlatIP(H) # コサイン類似度を計算するオブジェクトを生成
index.add(W_in_norm)   # コサイン類似度を計算するL2正規化された単語ベクトルを追加(V, H)
# 近似にしたいときは： index = faiss.IndexHNSWFlat(H, 32); index.hnsw.efSearch=64; index.add(W_in_norm)

# --- 4) CBOW文脈ベクトル作成 ---
def context_to_h(context_ids, W_in, weights=None):
    if weights is None:
        h = W_in[context_ids].mean(axis=0) # (100, ) コンテキストで渡したベクトルを列方向に足した後にコンテキスト数で割り、代表ベクトルを作る
    else:
        # 重み付けを行うときの処理
        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum()
        h = (W_in[context_ids] * w[:, None]).sum(axis=0)
    return h.astype(np.float32)

def predict_topk(context_ids, k=5, banned_ids=()):
    # 文脈平均 → 正規化
    h = context_to_h(context_ids, W_in)
    h_norm = h / (np.linalg.norm(h, ord=2) + 1e-12) # xと同じくコンテキストの代表ベクトルもL2ノルムで正規化
    x = h_norm[None, :]  # (H, )を(1, H) に次元を追加する

    # 検索（内積 = コサイン類似度） len(banned_ids)を足すのは、コンテキストが予測値に含まれるのを後で防ぐため
    sims, idxs = index.search(x, k + len(banned_ids))

    sims, idxs = sims[0].tolist(), idxs[0].tolist()

    # # 除外語（文脈語やUNKなど）を落とす
    banned = set(banned_ids)
    keep = [i for i in range(len(idxs)) if idxs[i] not in banned]
    idxs = [idxs[i] for i in keep][:k]
    sims = [sims[i] for i in keep][:k]

    # return [(id_to_word[idx], float(sim)) for idx, sim in zip(idxs, sims)]

# 例: 文 = "I like playing football", 中央: "like"
context_ids = [word_to_id["you"], word_to_id["toyota"], word_to_id["we"]]
# 文脈語は予測から除外
result = predict_topk(context_ids, k=5, banned_ids=context_ids)




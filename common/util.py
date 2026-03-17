# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *
import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def custom_preprocess(text_np):
    """各要素にテキストが入ったnumpy配列を受け取り、以下の値を返す
    return1: corpus 各要素の単語ID
    return2: word_to_id その単語のid
    return3: id_to_word その単語
    """
    corpus = []
    word_to_id = {}
    id_to_word = {}

    for text in text_np:
        if text not in word_to_id:
            new_id = len(id_to_word)
            word_to_id[text] = new_id
            id_to_word[new_id] = text
        corpus.append(word_to_id[text])
    
    return corpus, word_to_id, id_to_word


# def cos_similarity(x, y, eps=1e-8):
#     '''コサイン類似度の算出

#     :param x: ベクトル
#     :param y: ベクトル
#     :param eps: ”0割り”防止のための微小値
#     :return:
#     '''
#     nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
#     ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
#     return np.dot(nx, ny)

def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出
    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return
    '''

    nx = x / (np.sqrt(np.sum(x**2)) + eps) # xの正規化
    ny = y / (np.sqrt(np.sum(y**2)) + eps)# yの正規化

    return np.dot(nx, ny)


# def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
#     '''類似単語の検索

#     :param query: クエリ（テキスト）
#     :param word_to_id: 単語から単語IDへのディクショナリ
#     :param id_to_word: 単語IDから単語へのディクショナリ
#     :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
#     :param top: 上位何位まで表示するか
#     '''
#     if query not in word_to_id:
#         print('%s is not found' % query)
#         return

#     print('\n[query] ' + query)
#     query_id = word_to_id[query]
#     query_vec = word_matrix[query_id]

#     vocab_size = len(id_to_word)

#     similarity = np.zeros(vocab_size)
#     for i in range(vocab_size):
#         similarity[i] = cos_similarity(word_matrix[i], query_vec)

#     count = 0
#     for i in (-1 * similarity).argsort():
#         if id_to_word[i] == query:
#             continue
#         print(' %s: %s' % (id_to_word[i], similarity[i]))

#         count += 1
#         if count >= top:
#             return

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索

    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''

    if query not in word_to_id.keys():
        print(f"{query} is not found")
        return
    
    print("\n[query]" + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size =len(word_to_id)

    similarity = np.zeros(vocab_size)

    for i in range(vocab_size): 
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    count = 0

    for i in np.argsort(-1 * similarity): # 降順に並べ替えてインデックスを返すnp.argsort
        if id_to_word[i] == query:
            continue
        
        print(f"{id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


# def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def convert_one_hot(corpus:np.ndarray, vocab_size:int):

    # corpusのshapeに単語数を追加
    result_shape = corpus.shape + (vocab_size,)
    result = np.zeros(result_shape)

    # corpusが1次元の時
    if corpus.ndim == 1:
        for index, c in enumerate(corpus):
            for i in range(vocab_size):
                if c == i:
                    result[index, i] = 1
                    break

    elif corpus.ndim == 2:

        for index1, c_row in enumerate(corpus):
            for index2, c in enumerate(c_row):
                for i in range(vocab_size):
                    if c == i:
                        result[index1, index2, i] = 1
                        break
    return result    


# def create_co_matrix(corpus, vocab_size, window_size=1):
#     '''共起行列の作成

#     :param corpus: コーパス（単語IDのリスト）
#     :param vocab_size:語彙数
#     :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
#     :return: 共起行列
#     '''
#     corpus_size = len(corpus)
#     co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

#     for idx, word_id in enumerate(corpus):
#         for i in range(1, window_size + 1):
#             left_idx = idx - i
#             right_idx = idx + i

#             if left_idx >= 0:
#                 left_word_id = corpus[left_idx]
#                 co_matrix[word_id, left_word_id] += 1

#             if right_idx < corpus_size:
#                 right_word_id = corpus[right_idx]
#                 co_matrix[word_id, right_word_id] += 1

#     return co_matrix

def create_co_matrix(corpus, vocab_size, window_size=1):
    """共起行列の作成
    :param corpus: コーパス(単語のIDリスト)
    :param vocab_size: 語彙数
    :param window_size: 左右のコンテキストの大きさ
    :return 共起行列
    """

    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] =+ 1
    return co_matrix

# def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI（正の相互情報量）の作成

    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M

def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI（正の相互情報量）の作成
    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C) # スカラー(全要素の合計)
    row_sum = np.sum(C, axis=1) # ベクトル(列方向に合計) 各単語が別の単語と共起した回数
    col_sum = np.sum(C, axis=0) # ベクトル(行方向に合計) 各単語が別の単語と共起した回数
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (row_sum[i] * col_sum[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print(f"{100*cnt/total:.1f} done")
    return M


def create_contexts_target(corpus, window_size=1):
    '''コンテキストとターゲットの作成

    :param corpus: コーパス（単語IDのリスト）
    :param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

# def create_contexts_target(corpus, window_size=1):
#     target = corpus[window_size: -window_size]
#     contexts = []

#     for idx in range(window_size, len(corpus) - window_size):
#         cs = []
#         for t in range(-window_size, window_size + 1):
#             if t == 0:
#                 continue
#             cs.append(corpus[idx + t])
#         contexts.append(cs)
    
#     contexts = np.array(contexts)
#     target = np.array(target)

#     return contexts, target

def to_cpu(x):
    # import numpy
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss = 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    # 勾配の調整回数分
    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        # iterごとにバッチデータを作成するために開始位置をずらす(バッチデータを複数つくるため)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        # バッチデータの作成
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbose=False, is_reverse=False):
    correct = correct.flatten()
    # 頭の区切り文字
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 文字列へ変換
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbose:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0

    # word_matrix の各行ベクトルは事前に normalize() で単位ベクトル化していることを前提とする。
    # そのため query_vec を正規化してから行列積を取ることで、各単語ベクトルとのコサイン類似度に
    # 等しい値を一括で取得できる。逐次 cos_similarity() を呼び出すよりもループ処理と平方根計算を
    # 省けるため、探索処理を効率化できる。
def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

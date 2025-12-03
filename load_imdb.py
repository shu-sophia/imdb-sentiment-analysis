import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_imdb_data(vocab_size=10000, maxlen=200):
    """
    IMDbデータセットをロードし、NumPy配列として返す
    
    Parameters:
    -----------
    vocab_size : int
        使用する語彙数（頻出上位N語）
    maxlen : int
        シーケンスの最大長（短い場合は0でパディング、長い場合は切り捨て）
    
    Returns:
    --------
    (x_train, t_train), (x_test, t_test) : tuple of numpy arrays
        訓練データとテストデータ
        x_train, x_test: shape (samples, maxlen) の整数配列
        t_train, t_test: shape (samples,) のラベル配列（0 or 1）
    """
    # Kerasを使ってIMDbデータセットをダウンロード
    (x_train, t_train), (x_test, t_test) = imdb.load_data(num_words=vocab_size)
    
    # パディング処理（短い系列は0で埋め、長い系列は切り捨て）
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')
    
    # NumPy配列に変換（既にNumPy配列だが、明示的に変換）
    x_train = np.array(x_train, dtype=np.int32)
    x_test = np.array(x_test, dtype=np.int32)
    t_train = np.array(t_train, dtype=np.int32)
    t_test = np.array(t_test, dtype=np.int32)
    
    return (x_train, t_train), (x_test, t_test)


if __name__ == '__main__':
    # データをロード
    (x_train, t_train), (x_test, t_test) = load_imdb_data(vocab_size=10000, maxlen=200)
    
    # shape を確認
    print('訓練データ:')
    print(f'  x_train.shape: {x_train.shape}')
    print(f'  t_train.shape: {t_train.shape}')
    print()
    print('テストデータ:')
    print(f'  x_test.shape: {x_test.shape}')
    print(f'  t_test.shape: {t_test.shape}')
    print()
    print('データの例:')
    print(f'  x_train[0][:20]: {x_train[0][:20]}')
    print(f'  t_train[0]: {t_train[0]}')

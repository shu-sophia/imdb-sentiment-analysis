import numpy as np


def sigmoid(x):
    """シグモイド関数
    
    Parameters:
    -----------
    x : numpy.ndarray
        入力
    
    Returns:
    --------
    numpy.ndarray
        シグモイド関数の出力
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """ソフトマックス関数（オーバーフロー対策付き）
    
    Parameters:
    -----------
    x : numpy.ndarray
        入力 (N, D) または (D,)
    
    Returns:
    --------
    numpy.ndarray
        ソフトマックスの出力
    """
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        x = x - np.max(x)
        y = np.exp(x) / np.sum(np.exp(x))
    
    return y


def cross_entropy_error(y, t):
    """交差エントロピー誤差
    
    Parameters:
    -----------
    y : numpy.ndarray
        ニューラルネットワークの出力 (N, D)
    t : numpy.ndarray
        教師ラベル (N, D) または (N,)
    
    Returns:
    --------
    float
        交差エントロピー誤差
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 教師ラベルがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

import numpy as np
from common.functions import softmax, cross_entropy_error


class Sigmoid:
    """シグモイド活性化層"""
    
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        """順伝播
        
        Parameters:
        -----------
        x : numpy.ndarray
            入力
        
        Returns:
        --------
        numpy.ndarray
            シグモイド関数の出力
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        """逆伝播
        
        Parameters:
        -----------
        dout : numpy.ndarray
            上流からの勾配
        
        Returns:
        --------
        numpy.ndarray
            下流への勾配
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Softmax:
    """ソフトマックス層"""
    
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        """順伝播
        
        Parameters:
        -----------
        x : numpy.ndarray
            入力
        
        Returns:
        --------
        numpy.ndarray
            ソフトマックスの出力
        """
        self.out = softmax(x)
        return self.out
    
    def backward(self, dout):
        """逆伝播
        
        Parameters:
        -----------
        dout : numpy.ndarray
            上流からの勾配
        
        Returns:
        --------
        numpy.ndarray
            下流への勾配
        """
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    """ソフトマックス関数と交差エントロピー誤差を組み合わせた層"""
    
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル
        self.loss = None
    
    def forward(self, x, t):
        """順伝播
        
        Parameters:
        -----------
        x : numpy.ndarray
            入力 (N, D)
        t : numpy.ndarray
            教師ラベル (N,) または (N, D)
        
        Returns:
        --------
        float
            損失値
        """
        self.t = t
        self.y = softmax(x)
        
        # 教師ラベルがone-hotベクトルの場合、ラベルインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        """逆伝播
        
        Parameters:
        -----------
        dout : float or numpy.ndarray
            上流からの勾配（デフォルト1）
        
        Returns:
        --------
        numpy.ndarray
            下流への勾配
        """
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        
        return dx


class Affine:
    """全結合層（アフィン変換層）"""
    
    def __init__(self, W, b):
        """初期化
        
        Parameters:
        -----------
        W : numpy.ndarray
            重み行列 (入力次元, 出力次元)
        b : numpy.ndarray
            バイアス (出力次元,)
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        """順伝播
        
        Parameters:
        -----------
        x : numpy.ndarray
            入力 (N, D) または (N, T, D)
        
        Returns:
        --------
        numpy.ndarray
            出力 (N, H) または (N, T, H)
        """
        W, b = self.params
        self.x = x
        
        # 入力が3次元の場合（バッチサイズ、時系列、特徴量）
        self.original_shape = x.shape
        if x.ndim == 3:
            N, T, D = x.shape
            x = x.reshape(N * T, D)
        
        out = np.dot(x, W) + b
        
        # 元の形状に戻す
        if len(self.original_shape) == 3:
            N, T, D = self.original_shape
            out = out.reshape(N, T, -1)
        
        return out
    
    def backward(self, dout):
        """逆伝播
        
        Parameters:
        -----------
        dout : numpy.ndarray
            上流からの勾配
        
        Returns:
        --------
        numpy.ndarray
            下流への勾配
        """
        W, b = self.params
        
        # 入力が3次元だった場合の処理
        if len(self.original_shape) == 3:
            N, T, D = self.original_shape
            dout = dout.reshape(N * T, -1)
            x = self.x.reshape(N * T, D)
        else:
            x = self.x
        
        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        
        # 元の形状に戻す
        if len(self.original_shape) == 3:
            dx = dx.reshape(*self.original_shape)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx


class Embedding:
    """単語埋め込み層"""
    
    def __init__(self, W):
        """初期化
        
        Parameters:
        -----------
        W : numpy.ndarray
            重み行列（埋め込み行列） (vocab_size, embedding_dim)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        """順伝播
        
        Parameters:
        -----------
        idx : numpy.ndarray
            単語IDの配列 (N,) または (N, T)
        
        Returns:
        --------
        numpy.ndarray
            埋め込みベクトル (N, D) または (N, T, D)
        """
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        """逆伝播
        
        Parameters:
        -----------
        dout : numpy.ndarray
            上流からの勾配
        
        Returns:
        --------
        None (埋め込み層は勾配を下流に流さない)
        """
        dW, = self.grads
        dW[...] = 0  # 勾配を初期化
        
        # np.add.atを使って、同じインデックスへの勾配を累積
        np.add.at(dW, self.idx, dout)
        
        return None

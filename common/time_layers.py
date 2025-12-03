import numpy as np
from common.layers import Embedding


class TimeEmbedding:
    """時系列Embedding層"""
    
    def __init__(self, W):
        """初期化
        
        Parameters:
        -----------
        W : numpy.ndarray
            重み行列（埋め込み行列） (vocab_size, embedding_dim)
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    def forward(self, xs):
        """順伝播
        
        Parameters:
        -----------
        xs : numpy.ndarray
            単語IDのシーケンス (N, T)
        
        Returns:
        --------
        out : numpy.ndarray
            埋め込みベクトルのシーケンス (N, T, D)
        """
        N, T = xs.shape
        V, D = self.W.shape
        
        out = np.empty((N, T, D), dtype='f')
        self.layers = []
        
        # 各時刻でEmbeddingを適用
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        
        return out
    
    def backward(self, dout):
        """逆伝播
        
        Parameters:
        -----------
        dout : numpy.ndarray
            上流からの勾配 (N, T, D)
        
        Returns:
        --------
        None (Embedding層は勾配を下流に流さない)
        """
        N, T, D = dout.shape
        
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad
        
        return None

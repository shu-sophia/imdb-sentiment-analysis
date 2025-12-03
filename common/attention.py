import numpy as np
from common.functions import softmax


class AttentionLayer:
    """分類タスク用のAttention層
    
    最後の隠れ状態をQueryとして使用し、
    全時刻の隠れ状態から重要な部分に注目する
    """
    
    def __init__(self):
        self.params = []
        self.grads = []
        self.attention_weights = None
        self.hs = None
        self.h_last = None
    
    def forward(self, hs, h_last):
        """順伝播
        
        Parameters:
        -----------
        hs : numpy.ndarray
            TimeLSTMの全時刻の出力 (N, T, H)
        h_last : numpy.ndarray
            TimeLSTMの最後の隠れ状態 (N, H) - Queryとして使用
        
        Returns:
        --------
        c : numpy.ndarray
            コンテキストベクトル (N, H)
        """
        N, T, H = hs.shape
        
        # h_lastを(N, H, 1)に変形して、スコア計算
        # score[n, t] = hs[n, t, :] · h_last[n, :]
        # スコアの形状: (N, T)
        hr = h_last.reshape(N, H, 1)  # (N, H, 1)
        
        # バッチ乗算: (N, T, H) @ (N, H, 1) -> (N, T, 1)
        s = np.matmul(hs, hr)  # (N, T, 1)
        s = s.squeeze(axis=2)  # (N, T)
        
        # Softmaxで正規化
        a = softmax(s)  # (N, T)
        
        # 重み付き和: c = sum_t(a_t * h_t)
        # (N, T, 1) * (N, T, H) -> (N, T, H) -> sum over T -> (N, H)
        ar = a.reshape(N, T, 1)  # (N, T, 1)
        c = np.sum(ar * hs, axis=1)  # (N, H)
        
        # 逆伝播用にキャッシュ
        self.attention_weights = a
        self.hs = hs
        self.h_last = h_last
        
        return c
    
    def backward(self, dc):
        """逆伝播
        
        Parameters:
        -----------
        dc : numpy.ndarray
            コンテキストベクトルに関する勾配 (N, H)
        
        Returns:
        --------
        dhs : numpy.ndarray
            hsに関する勾配 (N, T, H)
        dh_last : numpy.ndarray
            h_lastに関する勾配 (N, H)
        """
        N, T, H = self.hs.shape
        a = self.attention_weights  # (N, T)
        hs = self.hs  # (N, T, H)
        h_last = self.h_last  # (N, H)
        
        # c = sum_t(a_t * h_t) の逆伝播
        # dc/da_t = h_t
        # dc/dh_t = a_t
        
        # (1) hsに関する勾配
        # dL/dhs[n,t,h] = a[n,t] * dc[n,h]
        dc_reshaped = dc.reshape(N, 1, H)  # (N, 1, H)
        a_reshaped = a.reshape(N, T, 1)  # (N, T, 1)
        dhs = a_reshaped * dc_reshaped  # (N, T, H)
        
        # (2) 重みaに関する勾配
        # dL/da[n,t] = sum_h(dc[n,h] * hs[n,t,h])
        da = np.sum(dc_reshaped * hs, axis=2)  # (N, T)
        
        # (3) Softmaxの逆伝播
        # ds = a * (da - sum(a * da))
        ds = a * da
        sum_ds = np.sum(ds, axis=1, keepdims=True)
        ds -= a * sum_ds
        
        # (4) スコアsに関する勾配からhs, h_lastへ
        # s[n,t] = hs[n,t,:] · h_last[n,:]
        # ds/dhs[n,t,h] = h_last[n,h]
        # ds/dh_last[n,h] = sum_t(hs[n,t,h] * ds[n,t])
        
        ds_reshaped = ds.reshape(N, T, 1)  # (N, T, 1)
        h_last_reshaped = h_last.reshape(N, 1, H)  # (N, 1, H)
        
        # hsへの勾配を追加
        dhs += ds_reshaped * h_last_reshaped  # (N, T, H)
        
        # h_lastへの勾配
        dh_last = np.sum(ds_reshaped * hs, axis=1)  # (N, H)
        
        return dhs, dh_last

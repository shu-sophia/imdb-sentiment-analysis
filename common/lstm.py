import numpy as np


class LSTM:
    """LSTM層（1ステップ分）"""
    
    def __init__(self, Wx, Wh, b):
        """初期化
        
        Parameters:
        -----------
        Wx : numpy.ndarray
            入力用の重み (D, 4H)
        Wh : numpy.ndarray
            隠れ状態用の重み (H, 4H)
        b : numpy.ndarray
            バイアス (4H,)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        """順伝播
        
        Parameters:
        -----------
        x : numpy.ndarray
            入力 (N, D)
        h_prev : numpy.ndarray
            前時刻の隠れ状態 (N, H)
        c_prev : numpy.ndarray
            前時刻のセル状態 (N, H)
        
        Returns:
        --------
        h_next : numpy.ndarray
            次時刻の隠れ状態 (N, H)
        c_next : numpy.ndarray
            次時刻のセル状態 (N, H)
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        # アフィン変換
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        
        # スライスして各ゲートを取得
        # f: 忘却ゲート, g: 新しい記憶, i: 入力ゲート, o: 出力ゲート
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        # 活性化関数を適用
        f = 1 / (1 + np.exp(-f))  # sigmoid
        g = np.tanh(g)
        i = 1 / (1 + np.exp(-i))  # sigmoid
        o = 1 / (1 + np.exp(-o))  # sigmoid
        
        # セル状態と隠れ状態の更新
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        
        # 逆伝播用にキャッシュ
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        """逆伝播
        
        Parameters:
        -----------
        dh_next : numpy.ndarray
            次時刻の隠れ状態に関する勾配 (N, H)
        dc_next : numpy.ndarray
            次時刻のセル状態に関する勾配 (N, H)
        
        Returns:
        --------
        dx : numpy.ndarray
            入力に関する勾配 (N, D)
        dh_prev : numpy.ndarray
            前時刻の隠れ状態に関する勾配 (N, H)
        dc_prev : numpy.ndarray
            前時刻のセル状態に関する勾配 (N, H)
        """
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        
        # tanh(c_next)の逆伝播
        tanh_c_next = np.tanh(c_next)
        
        # 出力ゲートの逆伝播
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        
        # セル状態の逆伝播
        dc_prev = ds * f
        
        # 各ゲートの逆伝播
        di = ds * g
        df = ds * c_prev
        dg = ds * i
        do = dh_next * tanh_c_next
        
        # sigmoidとtanhの逆伝播
        di *= i * (1 - i)
        df *= f * (1 - f)
        dg *= (1 - g ** 2)
        do *= o * (1 - o)
        
        # 4つのゲートを結合
        dA = np.hstack((df, dg, di, do))
        
        # アフィン変換の逆伝播
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)
        
        return dx, dh_prev, dc_prev


class TimeLSTM:
    """時系列LSTM層"""
    
    def __init__(self, Wx, Wh, b, stateful=False):
        """初期化
        
        Parameters:
        -----------
        Wx : numpy.ndarray
            入力用の重み (D, 4H)
        Wh : numpy.ndarray
            隠れ状態用の重み (H, 4H)
        b : numpy.ndarray
            バイアス (4H,)
        stateful : bool
            状態を保持するかどうか（今回はFalse固定でOK）
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h = None
        self.c = None
        self.dh = None
        self.stateful = stateful
    
    def forward(self, xs):
        """順伝播
        
        Parameters:
        -----------
        xs : numpy.ndarray
            入力シーケンス (N, T, D)
        
        Returns:
        --------
        hs : numpy.ndarray
            全時刻の隠れ状態 (N, T, H)
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        
        # LSTMレイヤを作成
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        
        # 初期の隠れ状態とセル状態（ゼロ初期化）
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
        
        # 各時刻でLSTMの順伝播を実行
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        """逆伝播
        
        Parameters:
        -----------
        dhs : numpy.ndarray
            全時刻の隠れ状態に関する勾配 (N, T, H)
        
        Returns:
        --------
        dxs : numpy.ndarray
            入力に関する勾配 (N, T, D)
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        dc = 0
        
        grads = [0, 0, 0]
        
        # 時刻を逆順に処理（BPTT）
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            
            # 勾配を累積
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # 勾配を保存
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh
        
        return dxs
    
    def set_state(self, h, c=None):
        """隠れ状態とセル状態を設定"""
        self.h = h
        self.c = c
    
    def reset_state(self):
        """隠れ状態とセル状態をリセット"""
        self.h = None
        self.c = None

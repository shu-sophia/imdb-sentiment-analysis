import numpy as np


class Adam:
    """Adam最適化アルゴリズム
    
    参考: https://arxiv.org/abs/1412.6980
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """初期化
        
        Parameters:
        -----------
        lr : float
            学習率
        beta1 : float
            1次モーメントの減衰率
        beta2 : float
            2次モーメントの減衰率
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None  # 1次モーメント
        self.v = None  # 2次モーメント
    
    def update(self, params, grads):
        """パラメータを更新
        
        Parameters:
        -----------
        params : list of numpy.ndarray
            パラメータのリスト
        grads : list of numpy.ndarray
            勾配のリスト
        """
        if self.m is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for i in range(len(params)):
            # 1次モーメントと2次モーメントを更新
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            # パラメータを更新
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

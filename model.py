"""
Attention機構を使った感情分析モデル
『ゼロから作るDeep Learning 2』スタイル
"""
import numpy as np
from common.layers import Affine, SoftmaxWithLoss
from common.time_layers import TimeEmbedding
from common.lstm import TimeLSTM
from common.attention import AttentionLayer


class AttentionClassifier:
    """Attention機構を使った感情分類モデル
    
    構成:
    1. TimeEmbedding: 単語ID → 埋め込みベクトル
    2. TimeLSTM: シーケンス処理
    3. AttentionLayer: 最後の隠れ状態をQueryとして使用
    4. Affine: コンテキストベクトル → クラススコア
    5. SoftmaxWithLoss: 損失計算
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """初期化
        
        Parameters:
        -----------
        vocab_size : int
            語彙数
        embedding_dim : int
            埋め込み次元
        hidden_dim : int
            LSTM隠れ状態の次元
        output_dim : int
            出力クラス数（2: positive/negative）
        """
        # 埋め込み層の重み（Xavier初期化）
        embed_W = (np.random.randn(vocab_size, embedding_dim) / 
                   np.sqrt(vocab_size)).astype('f')
        
        # LSTMの重み（Xavier初期化）
        lstm_Wx = (np.random.randn(embedding_dim, 4 * hidden_dim) / 
                   np.sqrt(embedding_dim)).astype('f')
        lstm_Wh = (np.random.randn(hidden_dim, 4 * hidden_dim) / 
                   np.sqrt(hidden_dim)).astype('f')
        lstm_b = np.zeros(4 * hidden_dim).astype('f')
        
        # Affine層の重み（Xavier初期化）
        affine_W = (np.random.randn(hidden_dim, output_dim) / 
                    np.sqrt(hidden_dim)).astype('f')
        affine_b = np.zeros(output_dim).astype('f')
        
        # レイヤの生成
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.attention = AttentionLayer()
        self.affine = Affine(affine_W, affine_b)
        self.loss_layer = SoftmaxWithLoss()
        
        # 全レイヤをリストにまとめる
        self.layers = [self.embed, self.lstm, self.attention, self.affine]
        
        # パラメータと勾配をまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, xs, ts):
        """順伝播
        
        Parameters:
        -----------
        xs : numpy.ndarray
            入力データ（単語IDのシーケンス） (N, T)
        ts : numpy.ndarray
            教師ラベル (N,)
        
        Returns:
        --------
        loss : float
            損失値
        """
        # 1. Embedding
        embed_out = self.embed.forward(xs)  # (N, T, D)
        
        # 2. LSTM（全時刻の隠れ状態を取得）
        hs = self.lstm.forward(embed_out)  # (N, T, H)
        
        # 3. Attention（最後の隠れ状態をQueryとして使用）
        h_last = hs[:, -1, :]  # (N, H)
        context = self.attention.forward(hs, h_last)  # (N, H)
        
        # 4. Affine（分類スコア）
        score = self.affine.forward(context)  # (N, output_dim)
        
        # 5. Softmax with Loss
        loss = self.loss_layer.forward(score, ts)
        
        return loss
    
    def backward(self, dout=1):
        """逆伝播
        
        Parameters:
        -----------
        dout : float
            上流からの勾配（通常は1）
        """
        # 5. SoftmaxWithLoss
        dscore = self.loss_layer.backward(dout)
        
        # 4. Affine
        dcontext = self.affine.backward(dscore)
        
        # 3. Attention
        dhs, dh_last = self.attention.backward(dcontext)
        
        # 注意: dh_lastは最後の隠れ状態への勾配なので、dhsに加算
        dhs[:, -1, :] += dh_last
        
        # 2. LSTM
        dembed = self.lstm.backward(dhs)
        
        # 1. Embedding
        self.embed.backward(dembed)
        
        return None
    
    def predict(self, xs):
        """予測
        
        Parameters:
        -----------
        xs : numpy.ndarray
            入力データ（単語IDのシーケンス） (N, T)
        
        Returns:
        --------
        predictions : numpy.ndarray
            予測ラベル (N,)
        """
        # 順伝播（損失計算なし）
        embed_out = self.embed.forward(xs)
        hs = self.lstm.forward(embed_out)
        h_last = hs[:, -1, :]
        context = self.attention.forward(hs, h_last)
        score = self.affine.forward(context)
        
        # 最大スコアのインデックスを予測ラベルとする
        predictions = np.argmax(score, axis=1)
        
        return predictions
    
    def accuracy(self, xs, ts):
        """正解率を計算
        
        Parameters:
        -----------
        xs : numpy.ndarray
            入力データ (N, T)
        ts : numpy.ndarray
            教師ラベル (N,)
        
        Returns:
        --------
        accuracy : float
            正解率（0.0～1.0）
        """
        predictions = self.predict(xs)
        accuracy = np.sum(predictions == ts) / len(ts)
        return accuracy

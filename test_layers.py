"""
NumPyベースのディープラーニングレイヤのテストスクリプト
『ゼロから作るDeep Learning 2』スタイルの実装を検証
"""
import numpy as np
import sys
import os

# commonモジュールをインポート
from common.functions import sigmoid, softmax, cross_entropy_error
from common.layers import Sigmoid, Softmax, SoftmaxWithLoss, Affine, Embedding
from common.lstm import LSTM, TimeLSTM
from common.time_layers import TimeEmbedding
from common.optimizer import Adam


def test_sigmoid():
    """Sigmoidレイヤのテスト"""
    print("=" * 50)
    print("Testing Sigmoid Layer...")
    print("=" * 50)
    
    layer = Sigmoid()
    x = np.array([[1.0, -0.5], [2.0, -1.0]])
    
    # 順伝播
    out = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output values:\n{out}")
    
    # 逆伝播
    dout = np.ones_like(out)
    dx = layer.backward(dout)
    print(f"Gradient shape: {dx.shape}")
    
    assert out.shape == x.shape, "Output shape mismatch"
    assert dx.shape == x.shape, "Gradient shape mismatch"
    print("✓ Sigmoid test passed!\n")


def test_softmax_with_loss():
    """SoftmaxWithLossレイヤのテスト"""
    print("=" * 50)
    print("Testing SoftmaxWithLoss Layer...")
    print("=" * 50)
    
    layer = SoftmaxWithLoss()
    x = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
    t = np.array([2, 0])  # 正解ラベル
    
    # 順伝播
    loss = layer.forward(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Loss: {loss}")
    
    # 逆伝播
    dx = layer.backward()
    print(f"Gradient shape: {dx.shape}")
    print(f"Gradient:\n{dx}")
    
    assert dx.shape == x.shape, "Gradient shape mismatch"
    print("✓ SoftmaxWithLoss test passed!\n")


def test_affine():
    """Affineレイヤのテスト"""
    print("=" * 50)
    print("Testing Affine Layer...")
    print("=" * 50)
    
    # 重みとバイアスの初期化
    D, H = 3, 5
    W = np.random.randn(D, H) * 0.01
    b = np.zeros(H)
    
    layer = Affine(W, b)
    x = np.random.randn(2, D)  # バッチサイズ2
    
    # 順伝播
    out = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Weight shape: {W.shape}")
    
    # 逆伝播
    dout = np.random.randn(*out.shape)
    dx = layer.backward(dout)
    print(f"Gradient shape: {dx.shape}")
    print(f"dW shape: {layer.grads[0].shape}")
    print(f"db shape: {layer.grads[1].shape}")
    
    assert out.shape == (2, H), "Output shape mismatch"
    assert dx.shape == x.shape, "Gradient shape mismatch"
    assert layer.grads[0].shape == W.shape, "Weight gradient shape mismatch"
    assert layer.grads[1].shape == b.shape, "Bias gradient shape mismatch"
    print("✓ Affine test passed!\n")


def test_embedding():
    """Embeddingレイヤのテスト"""
    print("=" * 50)
    print("Testing Embedding Layer...")
    print("=" * 50)
    
    # 埋め込み行列の初期化
    vocab_size, embedding_dim = 10, 5
    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    layer = Embedding(W)
    idx = np.array([1, 3, 5])  # 単語ID
    
    # 順伝播
    out = layer.forward(idx)
    print(f"Input (word IDs): {idx}")
    print(f"Output shape: {out.shape}")
    print(f"Embedding matrix shape: {W.shape}")
    
    # 逆伝播
    dout = np.random.randn(*out.shape)
    layer.backward(dout)
    print(f"Gradient shape: {layer.grads[0].shape}")
    
    assert out.shape == (3, embedding_dim), "Output shape mismatch"
    assert layer.grads[0].shape == W.shape, "Gradient shape mismatch"
    print("✓ Embedding test passed!\n")


def test_lstm():
    """LSTMレイヤのテスト"""
    print("=" * 50)
    print("Testing LSTM Layer...")
    print("=" * 50)
    
    # パラメータの初期化
    N, D, H = 2, 3, 4  # バッチサイズ、入力次元、隠れ状態次元
    Wx = np.random.randn(D, 4 * H) * 0.01
    Wh = np.random.randn(H, 4 * H) * 0.01
    b = np.zeros(4 * H)
    
    layer = LSTM(Wx, Wh, b)
    x = np.random.randn(N, D)
    h_prev = np.random.randn(N, H)
    c_prev = np.random.randn(N, H)
    
    # 順伝播
    h_next, c_next = layer.forward(x, h_prev, c_prev)
    print(f"Input shape: {x.shape}")
    print(f"h_prev shape: {h_prev.shape}")
    print(f"c_prev shape: {c_prev.shape}")
    print(f"h_next shape: {h_next.shape}")
    print(f"c_next shape: {c_next.shape}")
    
    # 逆伝播
    dh_next = np.random.randn(*h_next.shape)
    dc_next = np.random.randn(*c_next.shape)
    dx, dh_prev, dc_prev = layer.backward(dh_next, dc_next)
    print(f"dx shape: {dx.shape}")
    print(f"dh_prev shape: {dh_prev.shape}")
    print(f"dc_prev shape: {dc_prev.shape}")
    
    assert h_next.shape == (N, H), "h_next shape mismatch"
    assert c_next.shape == (N, H), "c_next shape mismatch"
    assert dx.shape == x.shape, "dx shape mismatch"
    print("✓ LSTM test passed!\n")


def test_time_lstm():
    """TimeLSTMレイヤのテスト"""
    print("=" * 50)
    print("Testing TimeLSTM Layer...")
    print("=" * 50)
    
    # パラメータの初期化
    N, T, D, H = 2, 5, 3, 4  # バッチサイズ、時系列長、入力次元、隠れ状態次元
    Wx = np.random.randn(D, 4 * H) * 0.01
    Wh = np.random.randn(H, 4 * H) * 0.01
    b = np.zeros(4 * H)
    
    layer = TimeLSTM(Wx, Wh, b)
    xs = np.random.randn(N, T, D)
    
    # 順伝播
    hs = layer.forward(xs)
    print(f"Input shape (xs): {xs.shape}")
    print(f"Output shape (hs): {hs.shape}")
    print(f"Expected output shape: ({N}, {T}, {H})")
    
    # 逆伝播
    dhs = np.random.randn(*hs.shape)
    dxs = layer.backward(dhs)
    print(f"Gradient shape (dxs): {dxs.shape}")
    print(f"Expected gradient shape: {xs.shape}")
    
    assert hs.shape == (N, T, H), f"Output shape mismatch: {hs.shape} != ({N}, {T}, {H})"
    assert dxs.shape == xs.shape, "Gradient shape mismatch"
    print("✓ TimeLSTM test passed!\n")


def test_time_embedding():
    """TimeEmbeddingレイヤのテスト"""
    print("=" * 50)
    print("Testing TimeEmbedding Layer...")
    print("=" * 50)
    
    # パラメータの初期化
    vocab_size, embedding_dim = 10, 5
    N, T = 2, 4  # バッチサイズ、時系列長
    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    layer = TimeEmbedding(W)
    xs = np.random.randint(0, vocab_size, (N, T))
    
    # 順伝播
    out = layer.forward(xs)
    print(f"Input shape (word IDs): {xs.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output shape: ({N}, {T}, {embedding_dim})")
    
    # 逆伝播
    dout = np.random.randn(*out.shape)
    layer.backward(dout)
    print(f"Gradient shape: {layer.grads[0].shape}")
    
    assert out.shape == (N, T, embedding_dim), "Output shape mismatch"
    assert layer.grads[0].shape == W.shape, "Gradient shape mismatch"
    print("✓ TimeEmbedding test passed!\n")


def test_adam():
    """Adam最適化のテスト"""
    print("=" * 50)
    print("Testing Adam Optimizer...")
    print("=" * 50)
    
    # パラメータと勾配の初期化
    params = [np.random.randn(3, 5), np.random.randn(5)]
    grads = [np.random.randn(3, 5), np.random.randn(5)]
    
    optimizer = Adam(lr=0.001)
    
    # 初期値を保存
    original_params = [p.copy() for p in params]
    
    # 更新
    optimizer.update(params, grads)
    
    print(f"Parameter 0 shape: {params[0].shape}")
    print(f"Parameter 1 shape: {params[1].shape}")
    print(f"Parameters updated: {not np.allclose(params[0], original_params[0])}")
    
    assert not np.allclose(params[0], original_params[0]), "Parameters not updated"
    print("✓ Adam optimizer test passed!\n")


def main():
    """すべてのテストを実行"""
    print("\n" + "=" * 50)
    print("NumPy Deep Learning Layers Test Suite")
    print("『ゼロから作るDeep Learning 2』スタイル")
    print("=" * 50 + "\n")
    
    try:
        test_sigmoid()
        test_softmax_with_loss()
        test_affine()
        test_embedding()
        test_lstm()
        test_time_lstm()
        test_time_embedding()
        test_adam()
        
        print("\n" + "=" * 50)
        print("✓✓✓ All tests passed! ✓✓✓")
        print("=" * 50 + "\n")
        
        print("Summary:")
        print("- Sigmoid layer: OK")
        print("- SoftmaxWithLoss layer: OK")
        print("- Affine layer: OK")
        print("- Embedding layer: OK")
        print("- LSTM layer: OK")
        print("- TimeLSTM layer: OK (output shape: (N, T, H))")
        print("- TimeEmbedding layer: OK")
        print("- Adam optimizer: OK")
        print("\nすべてのレイヤが正しく実装されています！")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

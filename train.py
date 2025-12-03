"""
Attention機構を使ったIMDb感情分析の学習スクリプト
『ゼロから作るDeep Learning 2』スタイル
"""
import numpy as np
import time
from load_imdb import load_imdb_data
from model import AttentionClassifier
from common.optimizer import Adam


def evaluate_accuracy(model, x_data, t_data, batch_size=100):
    """バッチごとに正解率を計算（メモリ効率化）
    
    Parameters:
    -----------
    model : AttentionClassifier
        評価するモデル
    x_data : numpy.ndarray
        入力データ
    t_data : numpy.ndarray
        教師ラベル
    batch_size : int
        バッチサイズ
    
    Returns:
    --------
    accuracy : float
        正解率
    """
    data_size = len(x_data)
    total_correct = 0
    
    for i in range(0, data_size, batch_size):
        batch_x = x_data[i:i+batch_size]
        batch_t = t_data[i:i+batch_size]
        predictions = model.predict(batch_x)
        total_correct += np.sum(predictions == batch_t)
    
    accuracy = total_correct / data_size
    return accuracy


def train():
    """学習のメイン関数"""
    
    # ハイパーパラメータ
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 128
    output_dim = 2  # positive/negative
    batch_size = 100
    epochs = 3
    learning_rate = 0.001
    maxlen = 200
    
    print("=" * 60)
    print("IMDb感情分析 - Attention機構による学習")
    print("=" * 60)
    print(f"ハイパーパラメータ:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  batch_size: {batch_size}")
    print(f"  epochs: {epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  maxlen: {maxlen}")
    print("=" * 60)
    
    # データの読み込み
    print("\nデータをロード中...")
    (x_train, t_train), (x_test, t_test) = load_imdb_data(
        vocab_size=vocab_size, 
        maxlen=maxlen
    )
    print(f"訓練データ: {x_train.shape}, ラベル: {t_train.shape}")
    print(f"テストデータ: {x_test.shape}, ラベル: {t_test.shape}")
    
    # モデルの初期化
    print("\nモデルを初期化中...")
    model = AttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # 最適化器の初期化
    optimizer = Adam(lr=learning_rate)
    
    # 学習ループ
    data_size = len(x_train)
    max_iters = data_size // batch_size
    
    print(f"\n学習開始 (全{epochs}エポック, 1エポックあたり{max_iters}イテレーション)")
    print("=" * 60)
    
    for epoch in range(epochs):
        # エポックごとの統計
        epoch_loss = 0
        start_time = time.time()
        
        # データをシャッフル
        idx = np.random.permutation(data_size)
        x_train_shuffled = x_train[idx]
        t_train_shuffled = t_train[idx]
        
        # ミニバッチ学習
        for iters in range(max_iters):
            # ミニバッチの取得
            batch_start = iters * batch_size
            batch_end = batch_start + batch_size
            batch_x = x_train_shuffled[batch_start:batch_end]
            batch_t = t_train_shuffled[batch_start:batch_end]
            
            # 順伝播と逆伝播
            loss = model.forward(batch_x, batch_t)
            model.backward()
            
            # パラメータの更新
            optimizer.update(model.params, model.grads)
            
            epoch_loss += loss
            
            # 進捗表示（10イテレーションごと）
            if (iters + 1) % 10 == 0:
                avg_loss = epoch_loss / (iters + 1)
                print(f"  Epoch {epoch+1}/{epochs}, Iter {iters+1}/{max_iters}, "
                      f"Loss: {avg_loss:.4f}", end='\r')
        
        # エポック終了後の評価
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / max_iters
        
        print()  # 改行
        
        # 訓練データでの正解率（サンプルを減らして高速化）
        print("  訓練データで評価中...", end=' ')
        train_sample_size = min(5000, len(x_train))
        train_sample_idx = np.random.choice(len(x_train), train_sample_size, replace=False)
        train_acc = evaluate_accuracy(
            model, 
            x_train[train_sample_idx], 
            t_train[train_sample_idx],
            batch_size=batch_size
        )
        print(f"完了")
        
        # テストデータでの正解率
        print("  テストデータで評価中...", end=' ')
        test_acc = evaluate_accuracy(
            model, 
            x_test, 
            t_test,
            batch_size=batch_size
        )
        print(f"完了")
        
        print(f"\nEpoch {epoch+1}/{epochs} 結果:")
        print(f"  時間: {epoch_time:.1f}秒")
        print(f"  訓練Loss: {avg_loss:.4f}")
        print(f"  訓練Accuracy: {train_acc*100:.2f}%")
        print(f"  テストAccuracy: {test_acc*100:.2f}%")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("学習完了！")
    print("=" * 60)
    
    # 最終評価
    print("\n最終評価:")
    final_train_acc = evaluate_accuracy(model, x_train, t_train, batch_size=batch_size)
    final_test_acc = evaluate_accuracy(model, x_test, t_test, batch_size=batch_size)
    print(f"  訓練データ正解率: {final_train_acc*100:.2f}%")
    print(f"  テストデータ正解率: {final_test_acc*100:.2f}%")
    
    # サンプル予測の表示
    print("\nサンプル予測:")
    sample_size = 10
    sample_idx = np.random.choice(len(x_test), sample_size, replace=False)
    sample_x = x_test[sample_idx]
    sample_t = t_test[sample_idx]
    sample_pred = model.predict(sample_x)
    
    label_names = {0: "Negative", 1: "Positive"}
    for i in range(sample_size):
        true_label = label_names[sample_t[i]]
        pred_label = label_names[sample_pred[i]]
        correct = "✓" if sample_t[i] == sample_pred[i] else "✗"
        print(f"  {i+1}. 正解: {true_label:8s} | 予測: {pred_label:8s} {correct}")


if __name__ == '__main__':
    train()

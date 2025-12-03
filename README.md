# IMDb Sentiment Analysis with Attention Mechanism

『ゼロから作るDeep Learning 2』スタイルのIMDb感情分析プロジェクト

## 概要

このプロジェクトは、**純粋なNumPy**でAttention機構を使った感情分析モデルを実装しています。IMDbのレビューデータを使用して、映画レビューが肯定的（Positive）か否定的（Negative）かを分類します。

TensorFlow/Kerasはデータのダウンロードのみに使用し、**すべてのモデル実装はNumPyのみ**で行われています。

## プロジェクト構成

```
imdb-sentiment-analysis/
├── load_imdb.py          # IMDbデータセットのローダー
├── model.py              # Attention機構を使った分類モデル
├── train.py              # 学習スクリプト
├── test_layers.py        # レイヤーのユニットテスト
└── common/               # 共通ライブラリ
    ├── layers.py         # 基本レイヤー（Affine, Sigmoid, SoftmaxWithLoss）
    ├── time_layers.py    # 時系列レイヤー（TimeEmbedding）
    ├── lstm.py           # LSTMレイヤー（LSTM, TimeLSTM）
    ├── attention.py      # Attention機構
    ├── optimizer.py      # 最適化器（Adam）
    └── functions.py      # 活性化関数など
```

## モデルアーキテクチャ

`AttentionClassifier`は以下の5つの層で構成されています：

1. **TimeEmbedding**: 単語ID → 埋め込みベクトル変換
2. **TimeLSTM**: シーケンス処理（全時刻の隠れ状態を出力）
3. **AttentionLayer**: 最後の隠れ状態をQueryとして使用し、重要な時刻に注目
4. **Affine**: コンテキストベクトル → クラススコア変換
5. **SoftmaxWithLoss**: 損失計算

```
入力 -> TimeEmbedding -> TimeLSTM -> AttentionLayer -> Affine -> SoftmaxWithLoss -> 損失
(単語ID)      ↓            ↓            ↓            ↓
          (埋め込み)   (隠れ状態)   (コンテキスト)  (スコア)
```

## セットアップ

このプロジェクトは `uv` を使用して依存関係を管理しています。

```bash
# 依存関係のインストール
uv sync
```

## 使用方法

### 1. データローダーのみ使用

```python
from load_imdb import load_imdb_data

# データのロード
(x_train, t_train), (x_test, t_test) = load_imdb_data(vocab_size=10000, maxlen=200)

print(x_train.shape)  # (25000, 200)
print(t_train.shape)  # (25000,)
```

### 2. モデルの学習

```bash
# 学習の実行
uv run python train.py
```

### 3. レイヤーのテスト

```bash
# ユニットテストの実行
uv run python test_layers.py
```

## ハイパーパラメータ

`train.py`で以下のハイパーパラメータを使用しています：

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `vocab_size` | 10,000 | 語彙数（頻出上位10,000語） |
| `embedding_dim` | 128 | 埋め込み次元 |
| `hidden_dim` | 128 | LSTM隠れ状態の次元 |
| `maxlen` | 200 | シーケンスの最大長 |
| `batch_size` | 100 | ミニバッチサイズ |
| `epochs` | 3 | エポック数 |
| `learning_rate` | 0.001 | 学習率（Adam） |

## 学習結果

3エポックの学習後の結果：

```
最終評価:
  訓練データ正解率: 94.32%
  テストデータ正解率: 83.70%
```

約16%のオーバーフィッティングが見られますが、純粋なNumPy実装で83.70%のテスト精度を達成しています。

## データ仕様

- **訓練データ**: 25,000件のレビュー
- **テストデータ**: 25,000件のレビュー
- **vocab_size**: 10,000（頻出上位10,000語）
- **maxlen**: 200（シーケンスの最大長）
  - 短い場合は0でパディング
  - 長い場合は切り捨て
- **ラベル**: 0（Negative）または 1（Positive）
- **データ形式**: 純粋なNumPy配列（`numpy.ndarray`）

## 依存関係

- NumPy >= 1.24.0
- TensorFlow >= 2.13.0（データのダウンロードのみ使用）
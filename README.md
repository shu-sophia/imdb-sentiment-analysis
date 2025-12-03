# IMDb Sentiment Analysis Data Loader

『ゼロから作るDeep Learning 2』スタイルのIMDbデータセット準備コード

## 概要

このプロジェクトは、感情分析（2値分類）のためのIMDbデータセットを準備する `load_imdb_data` 関数を提供します。TensorFlow/Kerasを使用してデータをダウンロードしますが、最終的な戻り値は**純粋なNumPy配列**です。

## セットアップ

このプロジェクトは `uv` を使用して依存関係を管理しています。

```bash
# 依存関係のインストール
uv sync

# スクリプトの実行
uv run python load_imdb.py
```

## 使用方法

```python
from load_imdb import load_imdb_data

# データのロード
(x_train, t_train), (x_test, t_test) = load_imdb_data(vocab_size=10000, maxlen=200)

print(x_train.shape)  # (25000, 200)
print(t_train.shape)  # (25000,)
print(x_test.shape)   # (25000, 200)
print(t_test.shape)   # (25000,)
```

## データ仕様

- **vocab_size**: 10,000 (頻出上位10,000語)
- **maxlen**: 200 (シーケンスの最大長)
  - 短い場合は0でパディング
  - 長い場合は切り捨て
- **ラベル**: 0 (ネガティブ) または 1 (ポジティブ)
- **データ形式**: 純粋なNumPy配列 (`numpy.ndarray`)

## 出力例

```
訓練データ:
  x_train.shape: (25000, 200)
  t_train.shape: (25000,)

テストデータ:
  x_test.shape: (25000, 200)
  t_test.shape: (25000,)

データの例:
  x_train[0][:20]: [   1   14   22   16   43  530  973 1622 1385   65  458 4468   66 3941
    4  173   36  256    5   25]
  t_train[0]: 1
```

## 依存関係

- NumPy >= 1.24.0
- TensorFlow >= 2.13.0 (データのダウンロードのみ使用)
# TYTAN
大規模QUBOアニーリングのためのSDK

## 問題形式
QUBO形式

## 問題サイズ
ローカル：1,000量子ビット程度
API経由：1,000-10万量子ビット程度

## インストール
```
pip install git+https://github.com/tytansdk/tytan
```

## 計算
```
from tytan import *

# QUBOを設定（今後dict形式に変更予定）
qubo = [[1.,1.],[1.,1.]]

# サンプラーを選択
sampler = sampler.SASampler()

# 計算
res = sampler.run(qubo, shots=100)

# 結果を出力
print(res)
```

# サンプラー
SASampler()

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

# 著作権
Copyright 2023 TYTAN TEAM

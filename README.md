# TYTAN（タイタン）
大規模QUBOアニーリングのためのSDK

## 問題形式
QUBO形式

## 問題サイズ
ローカル：1,000量子ビット程度   
API経由：1,000-10万量子ビット程度

## インストール
```
pip install tytan
```

```
pip install git+https://github.com/tytansdk/tytan
```

## 計算

### 事前準備

```
# 以下のコード例ではsympyが必要
pip install sympy
```

### コード例
```
from tytan import *
import sympy as sym

# 変数を定義
x, y, z = sym.symbols('x y z')

#式を記述
expr = 3*x**2 + 2*x*y + 4*y**2 + z**2 + 2*x*z + 2*y*z

# Compileクラスを使用して、QUBOを取得
qubo = qubo.Compile(expr).get_qubo()

# サンプラーを選択
sampler = sampler.SASampler()

# 計算
result = sampler.run(qubo, shots=100)
print(result)
```

### 出力例
```
[[{'z': 0, 'y': 0, 'x': 0}, 0.0, 8], [{'z': 1, 'y': 0, 'x': 0}, 1.0, 15], [{'z': 0, 'y': 0, 'x': 1}, 3.0, 12], [{'z': 0, 'y': 1, 'x': 0}, 4.0, 11], [{'z': 1, 'y': 0, 'x': 1}, 6.0, 17], [{'z': 1, 'y': 1, 'x': 0}, 7.0, 12], [{'z': 0, 'y': 1, 'x': 1}, 9.0, 16], [{'z': 1, 'y': 1, 'x': 1}, 14.0, 9]]
```

# サンプラー
SASampler()

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

# 著作権
Copyright 2023 TYTAN TEAM

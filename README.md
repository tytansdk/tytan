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

[[{'s1': 1, 's4': 1, 's3': 0, 's2': 1}, -196.0, 1.0],
 [{'s1': 0, 's4': 1, 's3': 1, 's2': 0}, -192.0, 1.0],
 [{'s1': 1, 's4': 0, 's3': 0, 's2': 1}, -192.0, 1.0],
 [{'s1': 0, 's4': 0, 's3': 1, 's2': 1}, -180.0, 1.0],
 [{'s1': 0, 's4': 1, 's3': 1, 's2': 1}, -160.0, 1.0],
 [{'s1': 0, 's4': 1, 's3': 0, 's2': 1}, -132.0, 1.0],
 [{'s1': 1, 's4': 1, 's3': 1, 's2': 0}, -96.0, 1.0],
 [{'s1': 1, 's4': 0, 's3': 1, 's2': 1}, -52.0, 1.0],
 [{'s1': 0, 's4': 0, 's3': 0, 's2': 0}, 0.0, 1.0],
 [{'s1': 1, 's4': 1, 's3': 1, 's2': 1}, 0.0, 1.0]]
```

# サンプラー
SASampler()

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

# 著作権
Copyright 2023 TYTAN TEAM

# **2023/5/26 大幅更新！記法が変わりました！**
**以下のサンプルコードは新しい記法に修正済みです。お手数ですがコードの修正をお願いします。**

**▼主な変更点**
- （sympy記法の場合）sympy.symbols() → tytan.symbols()　が推奨になりました（import sympyが不要）（ただし引き続き sympy.symbols() を使用可能）
- （すべての記法）tytan.qubo.Compile(expr).get_qubo() → tytan.Compile(expr).get_qubo()　になりました

# TYTAN（タイタン）
大規模QUBOアニーリングのためのSDKです。

**QUBO**を共通の入力形式とし、複数のサンプラーから選んでアニーリングできます。

## 問題サイズ
ローカル：1,000量子ビット程度

API経由：1,000-10万量子ビット程度

## インストール
更新が頻繁なためgithubからのインストールを推奨します。
```
pip install git+https://github.com/tytansdk/tytan
```

pipインストールはこちらです。
```
pip install tytan
```

## サンプルコード
基本的には定式化を行い、ソルバーと呼ばれる問題を解いてくれるプログラムにdict形式で入力をします。下記の例ではsympyを使って数式から入力をしています。数式以外にもcsvやnumpy matrixやpandas dataframeなど様々な入出力に対応しています。

sympy記法
```
#from tytan import symbols, Compile, sampler
from tytan import *

# 変数を定義
x, y, z = symbols('x y z')

#式を記述
expr = 3*x**2 + 2*x*y + 4*y**2 + z**2 + 2*x*z + 2*y*z

# Compileクラスを使用して、QUBOを取得
qubo, offset = Compile(expr).get_qubo()

# サンプラーを選択
solver = sampler.SASampler()

# 計算
result = solver.run(qubo, shots=100)
for r in result:
    print(r)
```

numpy記法
```
#from tytan import Compile, sampler
from tytan import *
import numpy as np

# QUBO行列を記述（上三角行列）
matrix = np.array([[3, 2, 2], [0, 4, 2], [0, 0, 2]])

# Compileクラスを使用して、QUBOを取得。
qubo, offset = Compile(matrix).get_qubo()

# サンプラーを選択
solver = sampler.SASampler()

# 計算
result = solver.run(qubo, shots=100)
for r in result:
    print(r)
```

pandas記法（csv読み込み）
```
#from tytan import Compile, sampler
from tytan import *
import pandas as pd

# QUBO行列を取得（csv, 上三角行列）
# header, index名を設定した場合は変数名が設定した名前になる。
csv_data = pd.read_csv('qubo.csv')

# Compileクラスを使用して、QUBOを取得
qubo, offset = Compile(csv_data).get_qubo()

# サンプラーを選択
solver = sampler.SASampler()

# 計算
result = solver.run(qubo, shots=100)
for r in result:
    print(r)
```

### 出力例
上記の例は入力に変数のラベルごと入力できるので、定式化した変数そのままで数値が戻ってきます。配列となっている答えは、「量子ビットの値」、「エネルギー（コスト）値」、「出現確率」の順で格納されています。

```
[{'z': 0, 'y': 0, 'x': 0}, 0.0, 8]
[{'z': 1, 'y': 0, 'x': 0}, 1.0, 15]
[{'z': 0, 'y': 0, 'x': 1}, 3.0, 12]
[{'z': 0, 'y': 1, 'x': 0}, 4.0, 11]
[{'z': 1, 'y': 0, 'x': 1}, 6.0, 17]
[{'z': 1, 'y': 1, 'x': 0}, 7.0, 12]
[{'z': 0, 'y': 1, 'x': 1}, 9.0, 16]
[{'z': 1, 'y': 1, 'x': 1}, 14.0, 9]
```

# サンプラー
サンプラーと呼ばれる計算をしてくれるプログラムが搭載されています。TYTANでは基本的には簡単なソルバーを用意はしていますが、ユーザーごとにソルバーを用意して簡単に繋げられるようにしています。ローカルのサンプラーのように手元のマシンにプログラムとして搭載する以外にも、APIのサンプラーでサーバーに接続する専用サンプラーなど様々な形態に対応できます。

ローカルサンプラー：
```
SASampler
GASampler
```

商用クラウドサンプラー：
```
ZekeSampler
NQSSampler
```

# 商用利用
TYTANは商用利用前提ですので、個人での利用はもちろん企業での活用を許可し、促進しています。

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

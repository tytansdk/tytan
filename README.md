# 最新情報
2023/06/10 便利なsymbols_list, symbols_define関数が追加されました。ドキュメントが作成されました。

2023/06/07 便利なAuto_arrayが追加されました。


# TYTAN（タイタン）
大規模QUBOアニーリングのためのSDKです。

QUBOを共通の入力形式とし、複数のサンプラーから選んでアニーリングできます。

入力は、数式を記述する方法、QUBO行列を入力する方法、QUBO行列をcsv読み込みする方法があります。

結果を自動で多次元配列に変換する機能を搭載。短いコードで確認できます。

## 問題サイズ
ローカルサンプラー：1,000量子ビット程度

API経由：1,000-10万量子ビット程度

## サンプラーと問題サイズ
基本的なローカルサンプラーの他、外部のAPIサンプラーなどを組み込めるようにしています。組み込みたソルバーがあればご連絡ください。

ローカルサンプラー：1,000量子ビット程度まで
```
SASampler
GASampler
```
商用クラウドサンプラー：1,000-100,000量子ビット程度
```
ZekeSampler
NQSSampler
```

## インストール
更新が頻繁なためgithubからのインストールを推奨します。
```
pip install git+https://github.com/tytansdk/tytan
```

pipインストールはこちらです。
```
pip install tytan
```

## サンプルコード１
3個の量子ビットのうち2個だけを1にする例です。数式を記述する方法、QUBO行列を入力する方法、QUBO行列をcsv読み込みする方法があります。

結果は「量子ビットの値」「エネルギー（コスト）値」「出現数」の順で格納されています。

▼数式を記述する方法
```python
from tytan import symbols, Compile, sampler

#量子ビットを用意
x = symbols('x')
y = symbols('y')
z = symbols('z')

#式を記述（3個のうち2個だけ1にする）
H = (x + y + z - 2)**2

#コンパイル
qubo, offset = Compile(H).get_qubo()

#サンプラー選択
solver = sampler.SASampler()

#サンプリング
result = solver.run(qubo, shots=100)

#結果
for r in result:
    print(r)
```
```
[{'x': 0.0, 'y': 1.0, 'z': 1.0}, -4.0, 24]
[{'x': 1.0, 'y': 0.0, 'z': 1.0}, -4.0, 23]
[{'x': 1.0, 'y': 1.0, 'z': 0.0}, -4.0, 53]
```

▼QUBO行列を入力する方法
```python
from tytan import symbols, Compile, sampler
import numpy as np

# QUBO行列を指定（上三角行列）
matrix = np.array([[-3, 2, 2], [0, -3, 2], [0, 0, -3]])
print(matrix)

#コンパイル
qubo, offset = Compile(matrix).get_qubo()

#サンプラー選択
solver = sampler.SASampler()

#サンプリング
result = solver.run(qubo, shots=100)

#すべての結果の確認
for r in result:
    print(r)
```
```
[[-3  2  2]
 [ 0 -3  2]
 [ 0  0 -3]]
[{'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, -4.0, 24]
[{'q0': 1.0, 'q1': 0.0, 'q2': 1.0}, -4.0, 23]
[{'q0': 1.0, 'q1': 1.0, 'q2': 0.0}, -4.0, 53]
```

▼QUBO行列をcsv読み込みする方法
```python
from tytan import symbols, Compile, sampler
import pandas as pd

# QUBO行列を読み込み（上三角行列）
# header, index名を設定すれば量子ビット名に反映される
matrix = pd.read_csv('matrix.csv', header=None)
print(matrix)

#コンパイル
qubo, offset = Compile(matrix).get_qubo()

#サンプラー選択
solver = sampler.SASampler()

#サンプリング
result = solver.run(qubo, shots=100)

#すべての結果の確認
for r in result:
    print(r)
```
```
   0  1  2
0 -3  2  2
1  0 -3  2
2  0  0 -3
[{'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, -4.0, 24]
[{'q0': 1.0, 'q1': 0.0, 'q2': 1.0}, -4.0, 26]
[{'q0': 1.0, 'q1': 1.0, 'q2': 0.0}, -4.0, 50]
```

### サンプルコード２
3ルーク問題は、3×3マスに3つのルーク（飛車）を互いに利きが及ばないように置く方法を探す問題です。二次元配列的な添字を持った量子ビットをまとめて定義する関数があります（配列形式も可）。サンプリングの乱数シードは固定できます。結果を二次元配列に戻して可視化する方法も3種類あります。詳しくは [ドキュメント](https://github.com/tytansdk/tytan/blob/main/document%20.md) を参照ください。

```python
from tytan import symbols_define, Compile, sampler, Auto_array

#量子ビットを用意（まとめて定義）
command = symbols_define([3, 3])
print(command)
exec(command)

#各行に1つだけ1
H = 0
H += (q0_0 + q0_1 + q0_2 - 1)**2
H += (q1_0 + q1_1 + q1_2 - 1)**2
H += (q2_0 + q2_1 + q2_2 - 1)**2

#各列に1つだけ1
H += (q0_0 + q1_0 + q2_0 - 1)**2
H += (q0_1 + q1_1 + q2_1 - 1)**2
H += (q0_2 + q1_2 + q2_2 - 1)**2

#コンパイル
qubo, offset = Compile(H).get_qubo()

#サンプラー選択（乱数シード固定）
solver = sampler.SASampler(seed=0)

#サンプリング（100回）
result = solver.run(qubo, shots=100)

#すべての結果を確認
print('result')
for r in result:
    print(r)

#1つ目の結果を自動配列で確認（ndarray形式）
arr, subs = Auto_array(result[0]).get_ndarray('q{}_{}')
print('get_ndarray')
print(arr)
print(subs)

#1つ目の結果を自動配列で確認（DataFrame形式）（1次元、2次元のみ）
df, subs = Auto_array(result[0]).get_dframe('q{}_{}')
print('get_dframe')
print(df)

#1つ目の結果を自動配列で確認（image形式）（2次元のみ）
img, subs = Auto_array(result[0]).get_image('q{}_{}')
import matplotlib.pyplot as plt
print('get_image')
plt.imshow(img)
plt.yticks(range(len(subs[0])), subs[0])
plt.xticks(range(len(subs[1])), subs[1])
plt.show()
```

```
q0_0 = symbols('q0_0')
q0_1 = symbols('q0_1')
q0_2 = symbols('q0_2')
q1_0 = symbols('q1_0')
q1_1 = symbols('q1_1')
q1_2 = symbols('q1_2')
q2_0 = symbols('q2_0')
q2_1 = symbols('q2_1')
q2_2 = symbols('q2_2')
result
[{'q0_0': 0.0, 'q0_1': 0.0, 'q0_2': 1.0, 'q1_0': 0.0, 'q1_1': 1.0, 'q1_2': 0.0, 'q2_0': 1.0, 'q2_1': 0.0, 'q2_2': 0.0}, -6.0, 19]
[{'q0_0': 0.0, 'q0_1': 0.0, 'q0_2': 1.0, 'q1_0': 1.0, 'q1_1': 0.0, 'q1_2': 0.0, 'q2_0': 0.0, 'q2_1': 1.0, 'q2_2': 0.0}, -6.0, 13]
[{'q0_0': 0.0, 'q0_1': 1.0, 'q0_2': 0.0, 'q1_0': 0.0, 'q1_1': 0.0, 'q1_2': 1.0, 'q2_0': 1.0, 'q2_1': 0.0, 'q2_2': 0.0}, -6.0, 21]
[{'q0_0': 0.0, 'q0_1': 1.0, 'q0_2': 0.0, 'q1_0': 1.0, 'q1_1': 0.0, 'q1_2': 0.0, 'q2_0': 0.0, 'q2_1': 0.0, 'q2_2': 1.0}, -6.0, 19]
[{'q0_0': 1.0, 'q0_1': 0.0, 'q0_2': 0.0, 'q1_0': 0.0, 'q1_1': 0.0, 'q1_2': 1.0, 'q2_0': 0.0, 'q2_1': 1.0, 'q2_2': 0.0}, -6.0, 11]
[{'q0_0': 1.0, 'q0_1': 0.0, 'q0_2': 0.0, 'q1_0': 0.0, 'q1_1': 1.0, 'q1_2': 0.0, 'q2_0': 0.0, 'q2_1': 0.0, 'q2_2': 1.0}, -6.0, 17]
get_ndarray
[[0 0 1]
 [0 1 0]
 [1 0 0]]
[['0', '1', '2'], ['0', '1', '2']]
get_dframe
   0  1  2
0  0  0  1
1  0  1  0
2  1  0  0
get_image
```
<img src="https://github.com/tytansdk/tytan/blob/main/img/img-01.png" width="%">


## 商用利用OK
TYTANは商用利用前提ですので、個人での利用はもちろん企業での活用を促進しています。

## 更新履歴
|日付|内容|
|:---|:---|
|2023/06/10|symbols_list, symbols_define追加、ドキュメント作成|
|2023/06/07|Auto_array追加|
|2023/06/01|シード固定追加|
|2023/05/26|全体構造修正|
|2023/03/28|SASampler高速化|
|2023/03/22|GASampler追加|
|2023/03/15|初期版|

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

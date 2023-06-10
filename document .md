## インストール（pip）
更新が頻繁なためgithubからのインストールを推奨します。
```
pip install -U git+https://github.com/tytansdk/tytan
```

pipインストールはこちらです。
```
pip install -U tytan
```

## import
コードの冒頭ですべての機能をインポートしておくことを推奨します。
```python
from tytan import symbols, symbols_list, symbols_define, Compile, sampler, Auto_array
```
または
```python
from tytan import *
```

## 量子ビットの定義（文字シンボルの定義）
基礎として、次のように文字シンボルを定義します。
```python
x = symbols('x')
```
```python
x, y, z = symbols('x y z')
```
```python
q = [symbols(f'q{i}') for i in range(5)] #[q0, q1, q2, q3, q4]
```

また、1次元～多次元の量子ビットを一度に定義できる関数が2種類あります。

文字シンボルを配列に定義する場合、次のようにndarray配列を得ます。
第2引数は省略するとデフォルトの添字形式になります。
```python
q = symbols_list([3, 3], 'q{}_{}')
print(q)
```
```
[[q0_0 q0_1 q0_2]
 [q1_0 q1_1 q1_2]
 [q2_0 q2_1 q2_2]]
```

文字シンボルを個別に定義する場合、次のように実行コマンドテキスト得てからexec()で実行します。
この方法では、IDEによっては後に構文警告（文字が未定義です）が表示されることがあります。
第2引数は省略するとデフォルトの添字形式になります。
```python
command = symbols_define([3, 3], 'q{}_{}')
print(command)
exec(command)
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
```

## 数式の設定
がんばります。

▼資料<br>
[TYTANチュートリアル](https://github.com/tytansdk/tytan_tutorial)<br>
[量子アニーリングのQUBOで設定可能な条件式まとめ（保存版）](https://vigne-cla.com/21-12/)（外部サイト）

▼勉強会<br>
[connpass「TYTAN」アニーリング](https://mdrft.connpass.com/)

▼コミュニティ<br>
[Discord「TYTAN」](https://discord.gg/qT5etstPW8)

## サンプリング
数式をコンパイルしてサンプリングする手続きはほぼこのままです。Hの部分が数式。サンプリングの乱数シードは固定可。サンプリング回数（shots）は適宜変更してください。
```python
#コンパイル
qubo, offset = Compile(H).get_qubo()

#サンプラー選択
solver = sampler.SASampler(seed=None)

#サンプリング
result = solver.run(qubo, shots=100)
```

▼サンプラー一覧<br>
ローカルサンプラー：1,000量子ビット程度まで
```python
SASampler()
GASampler()
```
商用クラウドサンプラー：1,000-100,000量子ビット程度　※要詳細
```python
ZekeSampler()
NQSSampler()
```

## 結果確認
結果はエネルギーが低い解から順にリスト形式で格納されています。「量子ビットの値」「エネルギー（コスト）値」「出現数」の順。
```python
for r in result:
    print(r)
```
```
[{'x': 0.0, 'y': 1.0, 'z': 1.0}, -4.0, 24]
[{'x': 1.0, 'y': 0.0, 'z': 1.0}, -4.0, 23]
[{'x': 1.0, 'y': 1.0, 'z': 0.0}, -4.0, 53]
```

0, 1のみを取り出す場合は次の通りだが、シンボルがアルファベット順のためq11がq2より手前に来たりすることに注意！（後述の方法でシンボルの自然順ソートが可能）
```python
for r in result:
    print(list(r[0].values()))
```
```
[0.0, 1.0, 1.0]
[1.0, 0.0, 1.0]
[1.0, 1.0, 0.0]
```

また、1次元～多次元の量子ビットの結果を見やすくする関数が3種類あります。いずれも結果リストから単一の結果を与えて使用します。

ndarray形式の配列に変換する方法は次のとおりです。1次元～5次元まで対応。1次元の場合でもこの関数を通すことでシンボルが自然順ソートされ、q11はq2より後に来るようになる。
```python
arr, subs = Auto_array(result[0]).get_ndarray('q{}_{}')
print(arr)
print(subs)
```
```
[[0 0 1]
 [0 1 0]
 [1 0 0]]
[['0', '1', '2'], ['0', '1', '2']]
```

DataFrame形式の配列に変換する方法は次のとおりです。1次元と2次元のみ対応。
```python
df, subs = Auto_array(result[0]).get_dframe('q{}_{}')
print(df)
```
```
   0  1  2
0  0  0  1
1  0  1  0
2  1  0  0
```

画像形式の配列に変換する方法は次のとおりです。2次元のみ対応。opencv形式、すなわちndarray(dtype='uint8')形式のグレースケール画像（0または255）です。
```python
img, subs = Auto_array(result[0]).get_image('q{}_{}')

import matplotlib.pyplot as plt
plt.imshow(img)
plt.yticks(range(len(subs[0])), subs[0])
plt.xticks(range(len(subs[1])), subs[1])
plt.show()
```
<img src="https://github.com/tytansdk/tytan/blob/main/img/img-01.png" width="%">



## QUBO行列を入力する方法
数式を記述する方法だけでなく、QUBO行列を入力する方法、QUBO行列をcsv読み込みする方法にも対応しています。それぞれのコンパイルまでの例です。サンプリング以降は共通です。

QUBO行列を入力する方法
```python
import numpy as np

# QUBO行列を指定（上三角行列）
matrix = np.array([[-3, 2, 2], [0, -3, 2], [0, 0, -3]])
print(matrix)

#コンパイル
qubo, offset = Compile(matrix).get_qubo()
```
```
[[-3  2  2]
 [ 0 -3  2]
 [ 0  0 -3]]
```

QUBO行列をcsv読み込みする方法
```python
import pandas as pd

# QUBO行列を読み込み（上三角行列）
# header, index名を設定すれば量子ビット名に反映される
matrix = pd.read_csv('matrix.csv', header=None)
print(matrix)

#コンパイル
qubo, offset = Compile(matrix).get_qubo()
```
```
   0  1  2
0 -3  2  2
1  0 -3  2
2  0  0 -3
```


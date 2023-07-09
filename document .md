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
from tytan import symbols, symbols_list, symbols_define, symbols_nbit, Compile, sampler, Auto_array
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
各自がんばってください。

▼資料<br>
[TYTANチュートリアル](https://github.com/tytansdk/tytan_tutorial)<br>
[量子アニーリングのQUBOで設定可能な条件式まとめ（保存版）](https://vigne-cla.com/21-12/)（外部サイト）

▼勉強会<br>
[connpass「TYTAN」アニーリング](https://mdrft.connpass.com/)

▼コミュニティ<br>
[Discord「TYTAN」](https://discord.gg/qT5etstPW8)

## サンプリング
数式をコンパイルしてサンプリングする手続きはほぼこのままです。Hの部分が数式。サンプリングの乱数シードは固定可。サンプリング回数（shots）はデフォルトが100なので適宜変更してください。
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
#SAサンプラー
solver = sampler.SASampler(seed=None)
result = solver.run(qubo, shots=100)
```
```python
#GAサンプラー
solver = sampler.GASampler(seed=None)
result = solver.run(qubo, shots=100)
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
[{'x': 0, 'y': 1, 'z': 1}, -4.0, 27]
[{'x': 1, 'y': 0, 'z': 1}, -4.0, 23]
[{'x': 1, 'y': 1, 'z': 0}, -4.0, 50]
```

0, 1のみを取り出す場合は次の通りだが、シンボルがアルファベット順のためq11がq2より手前に来たりすることに注意！（後述の方法でシンボルの自然順ソートが可能）
```python
for r in result:
    print(list(r[0].values()))
```
```
[0, 1, 1]
[1, 0, 1]
[1, 1, 0]
```

また、1次元～多次元の量子ビットの結果を見やすくする関数が3種類あります。いずれも結果リストから単一の結果を与えて使用します。

ndarray形式の配列に変換する方法は次のとおりです。1次元～5次元まで対応。1次元の場合でもこの関数を通すことでシンボルが自然順ソートされ、q11はq2より後に来るようになる（便利）。
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


## N-bitの変数を扱う方法
量子ビット（文字シンボル）はそのままでは0と1しか取り得ないため、幅広く整数や小数を表現するには複数の量子ビットを使ったN-bit表現を行う。例えば8-bitを使用して0から255までの整数を表現できる。これを簡単に扱うためのsymbols_nbit()関数とAuto_array().get_nbit_value()関数が用意されている。

例えば以下の連立方程式を解く場合、x, y, zとも0～255の整数であることが既知として、それぞれ8-bit表現して解くことができる。
10x+14y+4z = 5120<br>
9x+12y+2z = 4230<br>
7x+5y+2z = 2360

```python
#量子ビットをNビット表現で用意する
x = symbols_nbit(0, 256, 'x{}', num=8)
print(x)
y = symbols_nbit(0, 256, 'y{}', num=8)
z = symbols_nbit(0, 256, 'z{}', num=8)

#連立方程式の設定
H = 0
H += (10*x +14*y +4*z - 5120)**2
H += ( 9*x +12*y +2*z - 4230)**2
H += ( 7*x + 5*y +2*z - 2360)**2

#コンパイル
qubo, offset = Compile(H).get_qubo()
#サンプラー選択
solver = sampler.SASampler()
#サンプリング
result = solver.run(qubo, shots=500)

#１つ目の解をNビット表現から数値に戻して確認
print(result[0])
print('x =', Auto_array(result[0]).get_nbit_value(x))
print('y =', Auto_array(result[0]).get_nbit_value(y))
print('z =', Auto_array(result[0]).get_nbit_value(z))
```
```
128.0*x0 + 64.0*x1 + 32.0*x2 + 16.0*x3 + 8.0*x4 + 4.0*x5 + 2.0*x6 + 1.0*x7
[{'x0': 1, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 1, 'x7': 0, 'y0': 1, 'y1': 1, 'y2': 1, 'y3': 0, 'y4': 0, 'y5': 1, 'y6': 1, 'y7': 0, 'z0': 1, 'z1': 0, 'z2': 0, 'z3': 1, 'z4': 0, 'z5': 1, 'z6': 1, 'z7': 0}, -49676900.0, 2]
x = 130.0
y = 230.0
z = 150.0
```


## 解を網羅的に探索するオプション
使う人がいるか分からないが、複数の最適解を全部探索し尽くしたい場合のオプションを用意。

例えば1～5の数字を一列に並べるパターンは5!=120通りあるが、通常はすべての解を得るまでに多くのサンプリング数を必要とする。
```python
#量子ビットを用意する
q = symbols_list([5, 5], 'q{}_{}')

H = 0
#各スロットには数字が一つだけ入る（ワンホット）
for i in range(5):
    tmp = 0
    for k in range(5):
        tmp += q[i, k]
    H += (tmp - 1)**2

#各数字は一つのスロットに入る（ワンホット）
for k in range(5):
    tmp = 0
    for i in range(5):
        tmp += q[i, k]
    H += (tmp - 1)**2

#コンパイル
qubo, offset = Compile(H).get_qubo()

#サンプラー選択
solver = sampler.SASampler()

#サンプリング
result = solver.run(qubo, shots=500)

#最適解の種類をカウント
count = 0
for r in result:
    if r[1] == -offset:
        count += 1
print(count)
```
```
117
```
サンプリング数500では117種しか得られなかった。

そこで、次のオプションで解のバリエーションを優先したサンプリングが可能。ただし問題の規模が大きい場合は実行速度が遅い。
```python
result = solver.run(qubo, shots=500, countup=True) #show=Trueで進捗確認
```
```
120
```
120種類の解が得られた。

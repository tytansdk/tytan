## インストール（pip）
更新が頻繁なためgithubからのインストールを推奨します。
```
pip install git+https://github.com/tytansdk/tytan
```

pipインストールはこちらです。
```
pip install tytan
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
q = symbols_list([3, 3], format_txt='q{}_{}'))
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
command = symbols_define([3, 3], format_txt='q{}_{}')
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
▼資料<br>
[TYTANチュートリアル](https://github.com/tytansdk/tytan_tutorial)<br>
[量子アニーリングのQUBOで設定可能な条件式まとめ（保存版）](https://vigne-cla.com/21-12/#toc20)（外部サイト）

▼勉強会<br>
[connpass「TYTAN」アニーリング](https://mdrft.connpass.com/)

▼コミュニティ<br>
[Discord「TYTAN」](https://discord.gg/qT5etstPW8)

## サンプリング
数式をコンパイルしてサンプリングする手続きはほぼこのままです。サンプリング回数（shots）は適宜変更してください。
```python
#コンパイル
qubo, offset = Compile(H).get_qubo()

#サンプラー選択
solver = sampler.SASampler()

#サンプリング
result = solver.run(qubo, shots=100)
```

▼サンプラー一覧<br>
ローカルサンプラー：1,000量子ビット程度まで
```python
SASampler()
GASampler()
```
商用クラウドサンプラー：1,000-100,000量子ビット程度
```python
ZekeSampler()
NQSSampler()
```

## サンプルコード１
3個の量子ビットのうち2個だけを1にする例です。数式を記述する方法、QUBO行列を入力する方法、QUBO行列をcsv読み込みする方法があります。

結果は「量子ビットの値」「エネルギー（コスト）値」「出現数」の順で格納されています。

数式を記述
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

QUBO行列を入力
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

QUBO行列をcsv読み込み
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
3ルーク問題は、3×3マスに3つのルーク（飛車）を互いに利きが及ばないように置く方法を探す問題です。二次元配列的な問題では量子ビットに「q_0_0」「q0_a」「q(0)(0)」のように英数字の添え字を付けます。サンプリングの乱数シードは固定できます。結果はAuto_arrayクラスを使って自動的にソート、二次元配列化できます（3形式）。

```python
from tytan import symbols, Compile, sampler, Auto_array

#量子ビットを用意（まとめて指定）
q0_a, q0_b, q0_c = symbols('q0_a q0_b q0_c')
q1_a, q1_b, q1_c = symbols('q1_a q1_b q1_c')
q2_a, q2_b, q2_c = symbols('q2_a q2_b q2_c')

#各行に1つだけ1
H = 0
H += (q0_a + q0_b + q0_c - 1)**2
H += (q1_a + q1_b + q1_c - 1)**2
H += (q2_a + q2_b + q2_c - 1)**2

#各列に1つだけ1
H += (q0_a + q1_a + q2_a - 1)**2
H += (q0_b + q1_b + q2_b - 1)**2
H += (q0_c + q1_c + q2_c - 1)**2

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
import matplotlib.pyplot as plt
img, subs = Auto_array(result[0]).get_image('q{}_{}')
print('get_image')
plt.imshow(img)
plt.yticks(range(len(subs[0])), subs[0])
plt.xticks(range(len(subs[1])), subs[1])
plt.show()
```

```
result
[{'q0_a': 0.0, 'q0_b': 0.0, 'q0_c': 1.0, 'q1_a': 0.0, 'q1_b': 1.0, 'q1_c': 0.0, 'q2_a': 1.0, 'q2_b': 0.0, 'q2_c': 0.0}, -6.0, 19]
[{'q0_a': 0.0, 'q0_b': 0.0, 'q0_c': 1.0, 'q1_a': 1.0, 'q1_b': 0.0, 'q1_c': 0.0, 'q2_a': 0.0, 'q2_b': 1.0, 'q2_c': 0.0}, -6.0, 13]
[{'q0_a': 0.0, 'q0_b': 1.0, 'q0_c': 0.0, 'q1_a': 0.0, 'q1_b': 0.0, 'q1_c': 1.0, 'q2_a': 1.0, 'q2_b': 0.0, 'q2_c': 0.0}, -6.0, 21]
[{'q0_a': 0.0, 'q0_b': 1.0, 'q0_c': 0.0, 'q1_a': 1.0, 'q1_b': 0.0, 'q1_c': 0.0, 'q2_a': 0.0, 'q2_b': 0.0, 'q2_c': 1.0}, -6.0, 19]
[{'q0_a': 1.0, 'q0_b': 0.0, 'q0_c': 0.0, 'q1_a': 0.0, 'q1_b': 0.0, 'q1_c': 1.0, 'q2_a': 0.0, 'q2_b': 1.0, 'q2_c': 0.0}, -6.0, 11]
[{'q0_a': 1.0, 'q0_b': 0.0, 'q0_c': 0.0, 'q1_a': 0.0, 'q1_b': 1.0, 'q1_c': 0.0, 'q2_a': 0.0, 'q2_b': 0.0, 'q2_c': 1.0}, -6.0, 17]
get_ndarray
[[0 0 1]
 [0 1 0]
 [1 0 0]]
[['0', '1', '2'], ['a', 'b', 'c']]
get_dframe
   a  b  c
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
|2023/06/07|Auto_array追加|
|2023/06/01|シード固定追加|
|2023/05/26|全体構造修正|
|2023/03/28|SASampler高速化|
|2023/03/22|GASampler追加|
|2023/03/15|初期版|

# コントリビューター
https://github.com/tytansdk/tytan/graphs/contributors

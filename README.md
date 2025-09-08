## 最近のアップデート情報

★3次以上の項に対応したHOBOTANが統合されました。高次はget_hobo()とMIKASAmpler()を使用してください

★GPUで高速サンプリングができるArminSamplerが追加されました

★Compile()を100～300倍に高速化しました（derwind様提供）

全ての関数については [ドキュメント](https://github.com/tytansdk/tytan/blob/main/document%20.md) をご覧ください。

## チュートリアル

毎週木曜日22時からオンライン解説会。オンデマンド動画もあります。→ [TYTANチュートリアルのページ](https://github.com/tytansdk/tytan_tutorial)

## TYTAN（タイタン）
大規模QUBOアニーリングのためのSDKです。

QUBOを共通の入力形式とし、複数のサンプラーから選んでアニーリングできます。

入力は、数式を記述する方法、QUBO行列を入力する方法、QUBO行列をcsv読み込みする方法があります。

結果を自動で多次元配列に変換する機能を搭載。短いコードで確認できます。詳しくは [ドキュメント](https://github.com/tytansdk/tytan/blob/main/document%20.md) を参照ください。

## サンプラー
基本的なローカルサンプラーの他、外部のAPIサンプラーなどを組み込めるようにしています。組み込みたソルバーがあればご連絡ください。

QUBO（2次までの式）で使えるサンプラー
```
SASampler　：　ローカルCPU　★100量子ビットまではこれがおすすめ
GASampler　：　ローカルCPU
ArminSampler　：　ローカルCPU/GPU　★100量子ビット以上はこれがおすすめ
PieckSampler　：　ローカルCPU/GPU
ZekeSampler　：　商用クラウドサンプラー
NQSSampler　：　商用クラウドサンプラー
```

HOBO（3次以上の式）で使えるサンプラー
```
SASampler　：　ローカルCPU　★100量子ビットまではこれがおすすめ
ArminSampler　：　ローカルCPU/GPU
MIKASAmpler　：　ローカルCPU/GPU　★100量子ビット以上はこれがおすすめ
```


## インストール
対応するpythonバージョンは3.8以上です！

更新が頻繁なためgithubからのインストールを推奨します。
```
pip install -U git+https://github.com/tytansdk/tytan
```

pipインストールはこちらです。（更新忘れることがあります）
```
pip install -U tytan
```

## サンプルコード１
3個の量子ビットのうち2個だけを1にする例です。結果は「量子ビットの値」「エネルギー（コスト）値」「出現数」の順で格納されています。

```python
from tytan import *

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
result = solver.run(qubo)

#結果
for r in result:
    print(r)
```
```
[{'x': 0, 'y': 1, 'z': 1}, -4.0, 27]
[{'x': 1, 'y': 0, 'z': 1}, -4.0, 23]
[{'x': 1, 'y': 1, 'z': 0}, -4.0, 50]
```

## サンプルコード２
3ルーク問題は、3×3マスに3つのルーク（飛車）を互いに利きが及ばないように置く方法を探す問題です。量子ビットを2次元配列でまとめて定義する関数があります。サンプリングの乱数シードは固定できます。結果を二次元配列に戻して可視化する方法も3種類あります。詳しくは [ドキュメント](https://github.com/tytansdk/tytan/blob/main/document%20.md) を参照ください。

```python
from tytan import *

#量子ビットを用意（まとめて定義）
q = symbols_list([3, 3], 'q{}_{}')
print(q)

#各行に1つだけ1
H = 0
H += (q[0][0] + q[0][1] + q[0][2] - 1)**2
H += (q[1][0] + q[1][1] + q[1][2] - 1)**2
H += (q[2][0] + q[2][1] + q[2][2] - 1)**2

#各列に1つだけ1
H += (q[0][0] + q[1][0] + q[2][0] - 1)**2
H += (q[0][1] + q[1][1] + q[2][1] - 1)**2
H += (q[0][2] + q[1][2] + q[2][2] - 1)**2

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
[[q0_0 q0_1 q0_2]
 [q1_0 q1_1 q1_2]
 [q2_0 q2_1 q2_2]]
result
[{'q0_0': 0, 'q0_1': 0, 'q0_2': 1, 'q1_0': 0, 'q1_1': 1, 'q1_2': 0, 'q2_0': 1, 'q2_1': 0, 'q2_2': 0}, -6.0, 12]
[{'q0_0': 0, 'q0_1': 0, 'q0_2': 1, 'q1_0': 1, 'q1_1': 0, 'q1_2': 0, 'q2_0': 0, 'q2_1': 1, 'q2_2': 0}, -6.0, 16]
[{'q0_0': 0, 'q0_1': 1, 'q0_2': 0, 'q1_0': 0, 'q1_1': 0, 'q1_2': 1, 'q2_0': 1, 'q2_1': 0, 'q2_2': 0}, -6.0, 18]
[{'q0_0': 0, 'q0_1': 1, 'q0_2': 0, 'q1_0': 1, 'q1_1': 0, 'q1_2': 0, 'q2_0': 0, 'q2_1': 0, 'q2_2': 1}, -6.0, 27]
[{'q0_0': 1, 'q0_1': 0, 'q0_2': 0, 'q1_0': 0, 'q1_1': 0, 'q1_2': 1, 'q2_0': 0, 'q2_1': 1, 'q2_2': 0}, -6.0, 12]
[{'q0_0': 1, 'q0_1': 0, 'q0_2': 0, 'q1_0': 0, 'q1_1': 1, 'q1_2': 0, 'q2_0': 0, 'q2_1': 0, 'q2_2': 1}, -6.0, 15]
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


## サンプルコード３
3次以上の項を含む場合はQUBOではなくHOBOと呼ばれます。

事故を防ぐためコンパイル関数はQUBO用と区別しています。get_hobo()を使用してください。

サンプリング関数はArminSampler()でもMIKASAmpler()でもOK、中身同じです。区別したい方はQUBO:ArminSampler(), HOBO:MIKASAmpler()がおすすめ。

以下のコードは、5✕5の席にできるだけ多くの生徒を座らせる（ただし縦・横に3席連続で座ってはいけない）を解いたもの。

```python
import numpy as np
from tytan import *
import matplotlib.pyplot as plt

#量子ビットを用意
q = symbols_list([5, 5], 'q{}_{}')

#すべての席に座りたい（できれば）
H1 = 0
for i in range(5):
    for j in range(5):
        H1 += - q[i, j]

#どの直線に並ぶ3席も連続で座ってはいけない（絶対）
H2 = 0
for i in range(5):
    for j in range(5 - 3 + 1):
        H2 += np.prod(q[i, j:j+3])
for j in range(5):
    for i in range(5 - 3 + 1):
        H2 += np.prod(q[i:i+3, j])

#式の合体
H = H1 + 10*H2

#HOBOテンソルにコンパイル
hobo, offset = Compile(H).get_hobo()
print(f'offset\n{offset}')

#サンプラー選択
solver = sampler.MIKASAmpler()

#サンプリング
result = solver.run(hobo, shots=10000)

#上位3件
for r in result[:3]:
    print(f'Energy {r[1]}, Occurrence {r[2]}')

    #さくっと配列に
    arr, subs = Auto_array(r[0]).get_ndarray('q{}_{}')
    print(arr)

    #さくっと画像に
    img, subs = Auto_array(r[0]).get_image('q{}_{}')
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.show()
```
```
offset
0
MODE: GPU
DEVICE: cuda:0
Energy -17.0, Occurrence 686
[[1 1 0 1 1]
 [1 1 0 1 1]
 [0 0 1 0 0]
 [1 1 0 1 1]
 [1 1 0 1 1]]
Energy -17.0, Occurrence 622
[[1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]]
Energy -17.0, Occurrence 496
[[0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]]
```
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img1.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img2.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img3.png" width="%">


## 商用利用OK
TYTANは商用利用前提ですので、個人での利用はもちろん企業での活用を促進しています。

## 更新履歴
|日付|ver|内容|
|:---|:---|:---|
|2025/09/08|0.1.2|numpy2系での結果の表示を修正|
|2024/08/23|0.1.1|requirements.txtを軽量化|
|2024/08/12|0.1.0|HOBOTANを統合したメジャーアップデート、その他微修正|
|2024/08/12|0.0.30|Compile時の次数チェックの修正および高速化|
|2024/08/11|0.0.29|symbols_list, symbols_define関数をリファクタリング|
|2024/03/07|0.0.28|numpyのバージョン指定を解除|
|2024/02/25|0.0.27|PieckSampler追加（試験的）|
|2024/02/20|0.0.26|mps対応の修正|
|2024/02/18|0.0.25|symbols_list, symbols_define, symbols_nbitに関する修正|
|2024/02/13|0.0.23|ArminSamplerのデフォルトをGPUモードに, mps対応|
|2024/02/12|0.0.22|ArminSampler追加|
|2024/01/12|0.0.20|Compile高速化|
|2023/10/30|0.0.19|制約が2次を超える場合にエラーを返す|
|2023/08/31|0.0.18|安定版|
|2023/08/21|0.0.17|PyPI経由のインストールエラーを解消|
|2023/07/09|0.0.15|網羅探索するオプション追加|
|2023/07/03|0.0.14|Compile修正, requirements.txt修正|
|2023/06/21|0.0.12|Auto_array修正, SASampler性能UP|
|2023/06/17|0.0.9|requirements.txt修正, symbols_nbit, Auto_array.get_nbit_value追加|
|2023/06/12|0.0.8|SASampler性能UP, GASampler性能UP|
|2023/06/10|0.0.7|symbols_list, symbols_define追加、ドキュメント作成|
|2023/06/07|0.0.6|Auto_array追加|
|2023/06/01||シード固定追加|
|2023/05/26|0.0.5|全体構造修正|
|2023/03/28||SASampler高速化|
|2023/03/22||GASampler追加|
|2023/03/15||初期版|


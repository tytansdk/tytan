import re
import itertools
import numpy as np
import pandas as pd
from sympy import Symbol as sympy_Symbol

"""
量子ビットの添字を検出して結果を多次元配列に変換して返す

＜入力＞
result[0] : [dict_data, energy, occerrence]
もしくは
result[0][0] : dict_data

＜出力＞
ret : 多次元配列（存在しない要素は-1）
subs_sets : 1次元目から順に添字セット

＜呼び方例＞
arr, subs = Auto_array(result[0]).get_ndarray('q{}_{}')
df, subs = Auto_array(result[0]).get_dframe('q{}_{}')
img, subs = Auto_array(result[0]).get_image('q{}_{}')
添字を{}に置き換えること
"""

class Auto_array:
    def __init__(self, result0):
        #もし[dict_data, energy, occerrence]であればdict_dataを処理対象に
        if type(result0) is not dict:
            result0 = result0[0]
        
        self.a = result0
    
    #numpy形式で得る
    def get_ndarray(self, format_txt):
        
        #次元が1～5でなければエラー
        if format_txt.count('{}') not in [1, 2, 3, 4, 5]:
            raise
        
        a = self.a
        
        #全キー抽出
        keys = np.array(list(a.keys()))
        #print(keys)
        
        #フォーマット準備
        f = format_txt.replace('{}', '(.*)')
        #print(f)

        #フォーマットに従って添字抽出
        subs = [re.findall(f, key) for key in keys] #アンマッチは[]になっている
        subs = [sub for sub in subs if sub != []] #[]を削除
        subs = np.array(subs)
        #subs = np.array([re.findall(f, key) for key in keys])
        subs = np.squeeze(subs)
        if len(subs.shape) == 1:
            subs = subs[:, np.newaxis]
        #print(subs.shape)
        #print(subs)
        
        #添字の次元判定
        #(n, 1) -> 1次元
        #(n, 2) -> 2次元
        #(n, 3) -> 3次元
        dim = subs.shape[1]
        #print(dim)
        
        #次元ごとの添字セットを抽出、添字種類数も抽出
        subs_sets = []
        subs_dims = []
        for d in range(dim):
            #この次元の添字セット
            subs_set = np.array(list(set(subs[:, d])), str)
            #print(subs_set)
            
            try:
                #添字が数字として認識できれば
                [float(sub) for sub in subs_set]
                #print('float')
                
                #数字に従ってソート、自然順になる
                subs_float = np.array(subs_set, float)
                sorted_subs_set = subs_set[np.argsort(subs_float)]
            except:
                #添字に文字が一つでもあれば文字でソート
                sorted_subs_set = subs_set[np.argsort(subs_set)]
            #print(sorted_subs_set)
            
            #格納
            subs_sets.append(list(sorted_subs_set))
            subs_dims.append(len(sorted_subs_set))
        #print(subs_sets)
        #print(subs_dims)
        
        #行列を作成
        ret = np.ones(subs_dims, int) * -1
        
        #次元で分岐、面倒なのでとりあえずこれで5次元まで対応したこととする
        if dim == 1:
            for i, isub in enumerate(subs_sets[0]):
                try:
                    #あれば代入、なければ-1のまま
                    ret[i] = a[format_txt.format(isub)]
                except:
                    pass
            return ret, subs_sets[0]
        elif dim == 2:
            for (i, isub), (j, jsub) in itertools.product(enumerate(subs_sets[0]), enumerate(subs_sets[1])):
                try:
                    #あれば代入、なければ-1のまま
                    ret[i, j] = a[format_txt.format(isub, jsub)]
                except:
                    pass
        elif dim == 3:
            for (i, isub), (j, jsub), (k, ksub) in itertools.product(enumerate(subs_sets[0]), enumerate(subs_sets[1]), enumerate(subs_sets[2])):
                try:
                    #あれば代入、なければ-1のまま
                    ret[i, j, k] = a[format_txt.format(isub, jsub, ksub)]
                except:
                    pass
        elif dim == 4:
            for (i, isub), (j, jsub), (k, ksub), (l, lsub) in itertools.product(enumerate(subs_sets[0]), enumerate(subs_sets[1]), enumerate(subs_sets[2]), enumerate(subs_sets[3])):
                    try:
                        #あれば代入、なければ-1のまま
                        ret[i, j, k, l] = a[format_txt.format(isub, jsub, ksub, lsub)]
                    except:
                        pass
        elif dim == 5:
            for (i, isub), (j, jsub), (k, ksub), (l, lsub), (m, msub) in itertools.product(enumerate(subs_sets[0]), enumerate(subs_sets[1]), enumerate(subs_sets[2]), enumerate(subs_sets[3]), enumerate(subs_sets[4])):
                    try:
                        #あれば代入、なければ-1のまま
                        ret[i, j, k, l, m] = a[format_txt.format(isub, jsub, ksub, lsub, msub)]
                    except:
                        pass
        else:
            pass
        return ret, subs_sets
    
    
    #pandas形式で得る
    def get_dframe(self, format_txt):
        #次元が1か2でなければエラー
        if format_txt.count('{}') not in [1, 2]:
            raise
        
        #numpy形式
        arr, subs = self.get_ndarray(format_txt)
        
        #dframeを作成
        if format_txt.count('{}') == 1:
            arr = arr[:, np.newaxis].T
            df = pd.DataFrame(arr, columns=subs)
        else:
            df = pd.DataFrame(arr, columns=subs[1], index=subs[0])
        
        return df, subs
        
    #image形式で得る
    def get_image(self, format_txt):
        #次元が2でなければエラー
        if format_txt.count('{}') not in [2]:
            raise
        
        #numpy形式
        arr, subs = self.get_ndarray(format_txt)
        
        #imageを作成
        image = np.array(arr, 'uint8') * 255
        
        
        return image, subs
    
    #ｎbitを解析して値を得る
    def get_nbit_value(self, expr):
        
        #nbit式中のシンボル抽出
        symbols = list(expr.atoms(sympy_Symbol))
        #print(symbols)
        
        #nbit式中のシンボルに結果を戻す
        tmp = [(symbols[i], self.a[f'{symbols[i]}']) for i in range(len(symbols))]
        ans = expr.subs(tmp)
        
        #余計な少数0をなくして返す
        return float(ans)



    
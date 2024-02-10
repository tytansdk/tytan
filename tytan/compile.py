import re
import symengine
import numpy as np
import pandas as pd
from sympy import sympify, Poly


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


def degree_leq_2_check(expr):
    """
    項の次数が2以下かどうか調べる関数

    Args:
        expr: 項のsymengine表現
    Returns:
        bool: 次数が2以下かどうか
    """
    # 項が1変数だけならTrue
    if expr.is_Symbol:
        return True
    # 項が定数項ならTrue
    elif expr.is_Number:
        return True
    # 項がべき乗項のとき
    elif expr.is_Pow:
        # べき乗項に含まれる要素数が2より大きかったらFalse
        if len(expr.args) > 2:
            return False
        else:  # べき乗項が x^yで表されるとき
            base, exp = expr.args
            # 底が変数で指数が定数のとき
            if base.is_Symbol and exp.is_Number:
                # 指数が2以下ならTrue
                if int(exp) <= 2:
                    return True
                # 指数が3以上のときはFalse
                return False
            else:  # 指数が変数である場合などはFalse
                return False
    # 項が乗法のとき
    elif expr.is_Mul:
        # 要素数が3より大きい場合はFalse
        if len(expr.args) > 3:
            return False
        # 要素数が2以下ならTrue
        elif len(expr.args) <= 2:
            return True
        # 以下は要素数が3つの時
        # 3つすべてが変数の場合
        if all([arg.is_Symbol for arg in expr.args]):
            return False
        else:
            return True
    # 変数、定数、べき乗、乗法でないときFalse
    return False


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
            offset = 0
            for term in expr.args:

                # 定数項はオフセットに追加
                if term.is_Number:
                    offset += float(term)
                    continue

                if not degree_leq_2_check(term):  # 次数が2より大きい場合はエラー
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

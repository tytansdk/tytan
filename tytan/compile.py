import re
import symengine
import numpy as np
import pandas as pd


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

                if calc_degree(term) is None or calc_degree(term) > 2:  # 次数が2より大きい場合はエラー
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

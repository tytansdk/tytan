import sympy
import numpy as np
import pandas as pd

class Compile:
    def __init__(self, expr):
        self.expr = expr

    def get_qubo(self):
        """
        get qubo data
        Raises:
            TypeError: Input type is sympy, numpy or pandas.
        Returns:
            Tuple: qubo is dict. offset is float.
        """

        #sympy型のサブクラス
        if 'sympy.core' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = sympy.expand(self.expr)
            
            #二乗項を一乗項に変換
            expr = expr.replace(lambda e: isinstance(e, sympy.Pow) and e.exp == 2, lambda e: e.base)
            
            #もう一度同類項をまとめる
            expr = sympy.expand(expr)
            
            #定数項をoffsetとして抽出
            offset = expr.as_ordered_terms()[-1] #定数項は一番最後 #もう少し高速化できる？
            #定数項がなければ0
            if '*' in str(offset):
                offset = 0
            
            #offsetを引いて消す
            expr2 = expr - offset
            
            #文字と係数の辞書
            coeff_dict = expr2.as_coefficients_dict()
            
            #QUBO
            qubo = {}
            for key, value in coeff_dict.items():
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
            raise TypeError("Input type is sympy, numpy or pandas.")
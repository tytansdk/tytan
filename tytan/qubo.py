import sympy as sp
import numpy as np
import pandas as pd

class Compile:
    def __init__(self, expr):
        self.expr = expr

    def get_qubo(self):
        """get qubo data

        Raises:
            TypeError: Input type is sympy, numpy or pandas.

        Returns:
            Tuple: qubo is dict. offset is float.
        """

        # sympy
        if isinstance(self.expr, (sp.core.add.Add, sp.core.symbol.Symbol)):
            #展開されていない式は展開する
            self.expr = sp.expand(self.expr)

            # 式で使用されている変数を確認
            for item in self.expr.free_symbols:
                # バイナリなので、二次の項を一次の項に減らす代入
                self.expr = self.expr.subs([(item**2, item)])

            # 係数を抜き出す
            coeff_dict = self.expr.as_coefficients_dict()
            offset = coeff_dict.get(1, 0)

            # print(offset)

            expr2 = self.expr - offset

            coeff_dict = dict(expr2.as_coefficients_dict())

            # print(coeff_dict)

            # QUBOに格納開始
            qubo = {}
            for key, value in coeff_dict.items():
                # 一次の項の格納
                if key.count_ops() == 0:
                    qubo[(str(key), str(key))] = value

                # 二次の項の格納
                if key.count_ops() == 1:
                    arr = []
                    for term in key.args:
                        arr.append(term)
                    qubo[(str(arr[0]), str(arr[1]))] = value

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
                        qubo[(f'q{i}', f'q{j}')] = c

            return qubo, offset

        # pandas
        elif isinstance(self.expr, pd.core.frame.DataFrame):
            # 係数
            offset = 0

            # QUBOに格納開始
            qubo = {}
            for i, r in self.expr.iterrows():
                for j, c in enumerate(r):
                    if i <= j and c != 0:
                        if self.expr.columns.dtype == 'object':
                            qubo[(self.expr.columns[i], self.expr.columns[j])] = c
                        else:
                            qubo[(f'q{i}', f'q{j}')] = c

            return qubo, offset

        else:
            raise TypeError("Input type is sympy, numpy or pandas.")
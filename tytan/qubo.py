import sympy as sp

class Compile:
    def __init__(self, expr):
        self.expr = expr
    
    def get_qubo(self):
        
        #展開されていない式は展開する
        self.expr = sp.expand(self.expr)
        
        # 式で使用されている変数を確認
        for item in self.expr.free_symbols:
            # バイナリなので、二次の項を一次の項に減らす代入
            self.expr = self.expr.subs([(item**2, item)])

        # 係数を抜き出す
        coeff_dict = self.expr.as_coefficients_dict()
        offset = coeff_dict.get(1, 0)
        
        #print(offset)
        
        expr2 = self.expr - offset
        
        coeff_dict = dict(expr2.as_coefficients_dict())

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
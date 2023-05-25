from sympy import symbols as sympy_symbols

"""
SympyのSymbol関数にそのまま投げる関数
importをTYTANだけにするための申し訳ない方策
"""
def symbols(passed_txt):
    return sympy_symbols(passed_txt)
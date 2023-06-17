import numpy as np
import numpy.random as nr
import pandas as pd

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


#共通前処理
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
    
    def run(self, qubo, shots=10, T_num=8000):
        
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        #print(qmatrix)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 10)
        
        #アニーリングステップ数
        #T_num = 5000
        
        # --- SA ---
        
        # プール初期化
        pool_num = shots
        pool = nr.randint(0, 2, (pool_num, N))
        #print(pool)

        # スコア初期化
        score = np.array([q @ qmatrix @ q for q in pool])
        #print(score)
        
        #初期最大スコア差
        score_range = abs(np.max(score) - np.min(score))
        #print(score_range)
        
        #Tの下限
        T_min = 1.0 / score_range
        
        #温度配列　T_num
        #score_rangeが大きいほどT_minを小さくしないと終盤に落ち着かない
        Ts = (np.linspace(1, 0, num=T_num)**2) * (1 - T_min) + T_min
        #print(Ts)
        
        #フリップマスク　T_num, pool_num, N
        flip_mask = np.zeros((T_num*pool_num, N), bool)
        flip_mask[:, 0] = True
        flip_mask[:len(flip_mask)//2, 1] = True
        flip_mask = nr.default_rng((seed=self.seed).permuted(flip_mask, axis=1) #横シャッフル
        flip_mask = flip_mask.reshape(T_num, pool_num, N)
        #print(flip_mask.shape)
        
        #アニーリング
        for i, T in enumerate(Ts):
            #print(i, T)
            
            #フリップしたプール　pool_num, N
            pool_new = np.where(flip_mask[i], 1 - pool, pool)
            #print(pool_new)
            
            #スコア計算　pool_num
            score_new = np.array([q @ qmatrix @ q for q in pool_new])
            #print(score)
            #print(score_new)
            
            #採択確率　pool_num
            #np.abs(score_new - score) / score_range　-> 0～1　だが実際には 0～0.2　くらい
            #np.abs(score_new - score) / score_range / T　-> 0～5　とする
            #T -> 1～0.04
            #print(np.abs(score_new - score))
            Ps = 2.71828**(- np.abs(score_new - score) / score_range / T)
            #print(Ps)
            
            #更新
            update_mask = (score_new < score) + (nr.rand(pool_num) < Ps)
            #print(update_mask)
            pool[update_mask] = pool_new[update_mask]
            score[update_mask] = score_new[update_mask]
            
            # 進行表示1
            if i % 200 == 0:
                print("-", end="")
        print()
        
        # 局所探索フリップマスクリスト
        single_flip_mask = np.eye(N, dtype=bool)

        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            # フリップ後
            pool2 = np.where(fm, 1 - pool, pool)
            score2 = np.array([q @ qmatrix @ q for q in pool2])

            # 更新マスク
            update_mask = score2 < score

            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result


class GASampler:
    def __init__(self, seed=None):
        self.max_gen = 1000000
        self.max_count = 3
        self.seed = seed

    def run(self, qubo, shots=200):
        #共通前処理
        qmatrix, index_map, N = get_matrix(qubo)
        #print(qmatrix)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 200)

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
                print("-", end="")
                sw = False
                if gen % 10000 == 0:
                    print(" {}/{}".format(gen, self.max_gen))
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
            print()
        print("Automatic end at gen {}/{}".format(gen, self.max_gen))

        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result


class ZekeSampler:
    def __init__(self):
        self.API_ENDPOINT = "https://tytan-api.blueqat.com/v1/"
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

    def run(self, qubo, shots, api_key):
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

        result = self.create_task(data, api_key)
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
        qmatrix = np.zeros((N, N), dtype=object)

        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value

        qmatrix = np.insert(qmatrix, 0, keys, axis=1)
        qmatrix = np.vstack([[""] + keys[:], qmatrix])

        # print(qmatrix)

        # StringIOオブジェクトを作成する
        csv_buffer = io.StringIO()

        # CSVファイルの文字列を書き込む
        writer = csv.writer(csv_buffer)
        for row in qmatrix:
            writer.writerow(row)

        # CSVファイルの文字列を取得する
        Q = csv_buffer.getvalue()

        id = ulid.new()
        filename = "{}.zip".format(id.str)
        self.__to_zip(qubo=Q, filename=filename)
        files = {
            "zipfile": (
                filename,
                open("/tmp/{}".format(filename), "rb"),
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
        os.remove("/tmp/{}".format(filename))

        result = json.loads(r.text)
        return result

    def __to_zip(self, qubo: str, filename: str):
        import zipfile

        z_file = zipfile.ZipFile("/tmp/{}".format(filename), "w")
        z_file.writestr("qubo_w.csv", qubo, zipfile.ZIP_DEFLATED)
        z_file.close()

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
        qmatrix = np.zeros((N, N), dtype=object)

        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value

        qmatrix = np.insert(qmatrix, 0, keys, axis=1)
        qmatrix = np.vstack([[""] + keys[:], qmatrix])

        # print(qmatrix)

        # StringIOオブジェクトを作成する
        csv_buffer = io.StringIO()

        # CSVファイルの文字列を書き込む
        writer = csv.writer(csv_buffer)
        for row in qmatrix:
            writer.writerow(row)

        # CSVファイルの文字列を取得する
        Q = csv_buffer.getvalue()

        id = ulid.new()
        filename = "{}.zip".format(id.str)
        self.__to_zip(qubo=Q, filename=filename)
        files = {
            "zipfile": (
                filename,
                open("/tmp/{}".format(filename), "rb"),
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
        os.remove("/tmp/{}".format(filename))

        result = json.loads(r.text)
        return result

    def __to_zip(self, qubo: str, filename: str):
        import zipfile

        z_file = zipfile.ZipFile("/tmp/{}".format(filename), "w")
        z_file.writestr("qubo_w.csv", qubo, zipfile.ZIP_DEFLATED)
        z_file.close()









import numpy as np
import pandas as pd

class SASampler:
    def __init__(self):
        #アニーリングステップ数（後ろに局所探索が入ることに注意）
        #flip_count=1000, N=50, shots=100 -> 約１秒
        self.flip_step = 1000
    
    def run(self, qubo, shots):
        
        #start = time.time()
        
        #重複なしに要素を抽出
        keys = list(set(k for tup in qubo.keys() for k in tup))

        #要素のソート
        keys.sort()

        #抽出した要素のindexマップを作成
        index_map = {k: v for v, k in enumerate(keys)}

        #上記のindexマップを利用してタプルの内容をindexで書き換え
        qubo_index = {(index_map[k[0]], index_map[k[1]]): v for k, v in qubo.items()}

        #タプル内をソート
        qubo_sorted = {tuple(sorted(k)): v for k, v in sorted(qubo_index.items(), key=lambda x: x[1])}

        #量子ビット数
        N = int(len(keys))

        #QUBO matrix 初期化
        qmatrix = np.zeros((N, N))
        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value

        #df初期化
        columns = keys.copy()
        columns.append("energy")
        df = pd.DataFrame(columns=columns)

        
        
        #--- 高速疑似SA ---
        #プール初期化
        pool_num = shots
        pool = np.random.randint(0, 2, (pool_num, N))
        #重複は振り直し
        #パリエーションに余裕あれば確定非重複
        if pool_num < 2**(N - 1):
            #print('remake 1')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    while (pool[i] == pool[j]).all():
                        pool[j] = np.random.randint(0, 2, N)
        else:
            #パリエーションに余裕なければ3トライ重複可
            #print('remake 2')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    count = 0
                    while (pool[i] == pool[j]).all():
                        pool[j] = np.random.randint(0, 2, N)
                        count += 1
                        if count == 3:
                            break
        
        #スコア初期化
        score = np.array([q@qmatrix@q for q in pool])
        
        #フリップ数リスト（2個まで下がる）
        flip = np.sort(np.random.rand(self.flip_step)**2)[::-1]
        flip = (  flip * max(0, N*0.5 - 2)  ).astype(int) + 2
        #print(flip)
        
        #フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        for i in range(1, self.flip_step):
            tmp = [1] * flip[i] + [0] * (N - flip[i])
            np.random.shuffle(tmp)
            #前と重複なら振り直し
            while tmp == flip_mask[-1]:
                np.random.shuffle(tmp)
            flip_mask.append(tmp)
        flip_mask = np.array(flip_mask, bool)
        #print(flip_mask.shape)
        
        #局所探索フリップマスクリスト
        single_flip_mask = np.zeros((N, N), bool)
        for i in range(N):
            single_flip_mask[i, i] = True
        #print(single_flip_mask.shape)
        
        #end = time.time()
        #print('{}s'.format(round(end - start, 4)))
        
        #アニーリング
        #集団まるごと温度を下げる
        
        #start = time.time()
        
        for fm in flip_mask:
            #フリップ後
            pool2 = np.where(fm, 1 - pool, pool)
            score2 = np.array([q@qmatrix@q for q in pool2])
            
            #更新マスク
            update_mask = score2 < score
            
            #更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        #end = time.time()
        #print('{}s'.format(round(end - start, 4)))
        
        #start = time.time()
        
        #最後に1フリップ局所探索
        #集団まるごと
        for fm in single_flip_mask:
            #フリップ後
            pool2 = np.where(fm, 1 - pool, pool)
            score2 = np.array([q@qmatrix@q for q in pool2])
            
            #更新マスク
            update_mask = score2 < score
            
            #更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        #end = time.time()
        #print('{}s'.format(round(end - start, 4)))
        
        #後処理1
        expand = np.hstack((pool, score.reshape(pool_num, 1)))
        df = pd.DataFrame(expand, columns=df.columns)
        #----------

        #後処理2
        grouped = df.groupby(list(df.columns))
        counts = grouped.size().reset_index(name='occerrence')
        counts = counts.sort_values('energy').reset_index(drop=True)
        dict_data = counts.iloc[:,0:-2].to_dict(orient="records")
        result_dict = [[dict_data[index], row['energy'], int(row['occerrence'])] for index,row in counts.iterrows()]
 
        return result_dict

class GASampler:
    def __init__(self):
        self.max_gen = 1000000
        self.max_count = 2
    
    def run(self, qubo, shots):
        #重複なしに要素を抽出
        keys = list(set(k for tup in qubo.keys() for k in tup))

        #要素のソート
        keys.sort()

        #抽出した要素のindexマップを作成
        index_map = {k: v for v, k in enumerate(keys)}

        #上記のindexマップを利用してタプルの内容をindexで書き換え
        qubo_index = {(index_map[k[0]], index_map[k[1]]): v for k, v in qubo.items()}

        #タプル内をソート
        qubo_sorted = {tuple(sorted(k)): v for k, v in sorted(qubo_index.items(), key=lambda x: x[1])}

        #量子ビット数
        N = int(len(keys))

        #QUBO matrix 初期化
        qmatrix = np.zeros((N, N))
        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value

        #df初期化
        columns = keys.copy()
        columns.append("energy")
        df = pd.DataFrame(columns=columns)

        #--- GA ---
        #プール初期化
        pool_num = max(shots, 2) #N * 10
        pool = np.random.randint(0, 2, (pool_num, N))
        #スコア初期化
        score = np.array([q@qmatrix@q for q in pool])

        #進化
        last_mean_score = 99999
        best_score = np.copy(score)
        count = 0
        sw = True
        
        for gen in range(1, self.max_gen + 1):
            #親
            parent_id = np.random.choice(range(pool_num), 2, replace=False)
            parent = pool[parent_id]

            if N > 2:
                #交叉点
                cross_point = np.sort(np.random.choice(range(1, N), 2, replace=False))
                #家族
                c = np.array([parent[0],
                              parent[1],
                              np.hstack((parent[0, :cross_point[0]], parent[0, cross_point[0]:cross_point[1]], parent[1, cross_point[1]:])),
                              np.hstack((parent[0, :cross_point[0]], parent[1, cross_point[0]:cross_point[1]], parent[0, cross_point[1]:])),
                              np.hstack((parent[0, :cross_point[0]], parent[1, cross_point[0]:cross_point[1]], parent[1, cross_point[1]:])),
                              np.hstack((parent[1, :cross_point[0]], parent[0, cross_point[0]:cross_point[1]], parent[0, cross_point[1]:])),
                              np.hstack((parent[1, :cross_point[0]], parent[0, cross_point[0]:cross_point[1]], parent[1, cross_point[1]:])),
                              np.hstack((parent[1, :cross_point[0]], parent[1, cross_point[0]:cross_point[1]], parent[0, cross_point[1]:]))])
                #評価
                s = np.array([c[0]@qmatrix@c[0], c[1]@qmatrix@c[1], c[2]@qmatrix@c[2], c[3]@qmatrix@c[3], c[4]@qmatrix@c[4], c[5]@qmatrix@c[5]])
            
            elif N == 2:
                #家族
                c = np.array([parent[0],
                              parent[1],
                              [parent[0, 0], parent[1, 1]],
                              [parent[0, 1], parent[1, 0]]])
                #評価
                s = np.array([c[0]@qmatrix@c[0], c[1]@qmatrix@c[1], c[2]@qmatrix@c[2], c[3]@qmatrix@c[3]])
            
            elif N == 1:
                #家族
                c = np.array([parent[0],
                              parent[1],
                              1 - parent[0],
                              1 - parent[1]])
                #評価
                s = np.array([c[0]@qmatrix@c[0], c[1]@qmatrix@c[1], c[2]@qmatrix@c[2], c[3]@qmatrix@c[3]])

            #エリート選択
            select_id = np.argsort(s)[:2]
            #交代
            pool[parent_id] = c[select_id]
            score[parent_id] = s[select_id]
            #進行表示1
            if gen % 500 == 0:
                print('-', end='')
                sw = False
                if gen % 10000 == 0:
                    print(' {}/{}'.format(gen, self.max_gen))
                    sw = True
            #終了判定
            if gen % 10 == 0:
                if np.sum(score - best_score) == 0:
                    count += 1
                else:
                    best_score = np.copy(score)
                    count = 0
                if count >= self.max_count:
                    break
        #進行表示2
        if not sw: print()
        print('Automatic end at gen {}/{}'.format(gen, self.max_gen))

        #後処理1
        expand = np.hstack((pool, score.reshape(pool_num, 1)))
        df = pd.DataFrame(expand, columns=df.columns)
        #----------

        #後処理2
        grouped = df.groupby(list(df.columns))
        counts = grouped.size().reset_index(name='occerrence')
        counts = counts.sort_values('energy').reset_index(drop=True)
        dict_data = counts.iloc[:,0:-2].to_dict(orient="records")
        result_dict = [[dict_data[index], row['energy'], int(row['occerrence'])] for index,row in counts.iterrows()]
 
        return result_dict

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
        
        #重複なしに要素を抽出
        keys = list(set(k for tup in qubo.keys() for k in tup))
        #print(keys)

        #抽出した要素のindexマップを作成
        index_map = {k: v for v, k in enumerate(keys)}
        #print(index_map)

        #上記のindexマップを利用してタプルの内容をindexで書き換え
        qubo_index = {(index_map[k[0]], index_map[k[1]]): v for k, v in qubo.items()}
        #print(qubo_index)

        #タプル内をソート
        qubo_sorted = {tuple(sorted(k)): v for k, v in sorted(qubo_index.items(), key=lambda x: x[1])}
        #print(qubo_sorted)

        #量子ビット数
        N = int(len(keys))

        #QUBO matrix 初期化
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

        #print(qmatrix)
        #print(variable_labels)
        #print(linear_biases)
        #print(quadratic_biases)
        #print(quadratic_head)
        #print(quadratic_tail)
        
        num_interactions = len(quadratic_biases)
        
        # クラウドにpostするBQM
        bqm = {'type': 'BinaryQuadraticModel',
               'version': {'bqm_schema': '3.0.0'},
               'use_bytes': False,
               'index_type': 'int32',
               'bias_type': 'float64',
               'num_variables': N,
               'num_interactions': num_interactions,
               'variable_labels': list(variable_labels),
               'variable_type': 'BINARY',
               'offset': 0.0,
               'info': {},
               'linear_biases': list(linear_biases),
               'quadratic_biases': quadratic_biases,
               'quadratic_head': quadratic_head,
               'quadratic_tail': quadratic_tail}
        
        data = {
            "bqm": bqm,
            "shots": shots,
        }
        
        result = self.create_task(data, api_key)
        #print(result)
        
        #エネルギーを取り出し
        energy_list = result["result"]["vectors"]["energy"]["data"]
        #print(energy_list)
        
        #出現回数を取り出し
        occurrences_list = result["result"]["vectors"]["num_occurrences"]["data"]
        #print(occurrences_list)
        
        #サンプルをリストとして取り出し
        num_digits = result["result"]["num_variables"]
        sample_data = result["result"]["sample_data"]["data"]
        #print(sample_data)

        binary_str = [format(i[0], f'0{num_digits}b') for i in sample_data]
        binary_list = [[int(bit) for bit in reversed(i)] for i in binary_str]
        #print(binary_list)
        
        variable_labels = result["result"]["variable_labels"]
        #print(variable_labels)
        
        result_list = []
        for index, item in enumerate(binary_list):
            result_list.append([{k: v for k, v in zip(variable_labels, item)}, energy_list[index], occurrences_list[index]])
        
        #print(result_list)
        
        return result_list
    
class NQSSampler:

    from typing import Optional

    def __init__(self, api_key: Optional[str] = None):
        self.API_ENDPOINT = "https://tytan-api.blueqat.com/v1/"
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


    def __get_headers(self, api_key: Optional[str]) -> dict[str, str]:
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
    
    def __init__(self, api_endpoint:str="http://localhost:8080/ngqs/v1/solve") -> None:
        self.API_ENDPOINT = api_endpoint
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

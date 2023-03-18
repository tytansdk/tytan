import numpy as np
import pandas as pd

class SASampler:
    def __init__(self):
        #: Initial Temperature
        self.Ts = 5
        #: Final Temperature
        self.Tf = 0.02

        #: Descreasing rate of temperature
        self.R = 0.9
        #: Iterations
        self.ite = 100

    def run(self, qubo, shots):
        
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

        for (i, j), value in qubo_sorted.items():
            qmatrix[i, j] = value
        #print(qmatrix)

        columns = keys.copy()
        columns.append("energy")
        df = pd.DataFrame(columns=columns)
        
        for i in range(shots):
            T = self.Ts
            q = np.random.choice([0,1], N)
            while T > self.Tf:
                x_list = np.random.randint(0, N, self.ite)
                for x in x_list:
                    q_copy = np.ones(N)*q[x]
                    q_copy[x] = 1-q_copy[x]
                    dE = -2*sum(q*q_copy*qmatrix[:,x])

                    if dE < 0 or np.exp(-dE/T) > np.random.random_sample():
                        q[x] = 1 - q[x]
                T *= self.R

            new_row_df = pd.DataFrame([list(np.append(q, 0))], columns=df.columns)
            new_row_df['energy'] = q@qmatrix@q
            df = pd.concat([df, new_row_df], ignore_index=True)    
    
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
        import urllib
        
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
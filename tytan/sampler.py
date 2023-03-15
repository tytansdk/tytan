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
        counts = grouped.size().reset_index(name='occerence')
        counts = counts.sort_values('energy').reset_index(drop=True)
        dict_data = counts.iloc[:,0:-2].to_dict(orient="records")
        result_dict = [[dict_data[index], row['energy'], int(row['occerence'])] for index,row in counts.iterrows()]
 
        return result_dict
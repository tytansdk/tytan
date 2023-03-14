import numpy as np

class SASampler:
    def __init__(self):
        #: Initial Temperature
        self.Ts = 2
        #: Final Temperature
        self.Tf = 0.02

        #: Descreasing rate of temperature
        self.R = 0.85
        #: Iterations
        self.ite = 50

    def run(self, qubo, shots):
        
        qubo = np.triu(qubo)
        N = len(qubo)

        result = []
        for i in range(shots):
            T = self.Ts
            q = np.random.choice([0,1], N)
            while T > self.Tf:
                x_list = np.random.randint(0, N, self.ite)
                for x in x_list:
                    q_copy = np.ones(N)*q[x]
                    q_copy[x] = 1
                    dE = -2*sum(q*q_copy*qubo[:,x])

                    if np.exp(-dE/T) > np.random.random_sample():
                        q[x] = 1 - q[x]
                T *= self.R
            result.append((list(q), q@qubo@q))
        return result
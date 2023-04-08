import sympy as sym

from tytan import qubo
from tytan.sampler import NQSLocalSampler

x, y, z = sym.symbols("x y z")
expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
Q, _offset = qubo.Compile(expr).get_qubo()
sampler = NQSLocalSampler()
result = sampler.run(qubo=Q)
print(result)
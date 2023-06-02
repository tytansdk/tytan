import sympy as sym

from tytan import qubo
from tytan.sampler import GASampler


def test_ga_sampler_run():
    x, y, z = sym.symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    Q, _offset = qubo.Compile(expr).get_qubo()
    sampler = GASampler()
    result = sampler.run(qubo=Q, shots=1)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] is not None
    assert result[0][0]["y"] is not None
    assert result[0][0]["z"] is not None
    assert result[0][1] is not None
    assert result[0][2] is not None

def test_ga_sampler_run_with_seed():
    x, y, z = sym.symbols("x y z")
    expr = (x - y) ** 2 + (y - z) ** 2
    Q, _offset = qubo.Compile(expr).get_qubo()
    sampler = GASampler(seed=0)
    result = sampler.run(qubo=Q, shots=1)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] == 0
    assert result[0][0]["y"] == 1
    assert result[0][0]["z"] == 1

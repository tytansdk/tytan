import os
import pytest
import sympy as sym

from tytan import qubo
from tytan.sampler import NQSSampler

def test_nqs_sampler_run():
    x, y, z = sym.symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    Q, _offset = qubo.Compile(expr).get_qubo()
    api_key = os.environ.get("TYTAN_API_KEY", "foobar")
    sampler = NQSSampler(api_key)
    result = sampler.run(qubo=Q)
    assert result is not None
    assert result["result"] is not None
    assert result["result"]["x"] == 0
    assert result["result"]["y"] == 0
    assert result["result"]["z"] == 0
    assert result["energy"] == 0
    assert result["time"] is not None

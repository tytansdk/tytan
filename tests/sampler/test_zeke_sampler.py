import os
import pytest
import sympy as sym

from tytan import qubo
from tytan.sampler import ZekeSampler


@pytest.mark.vcr(filter_headers=["x-api-key"])
def test_zeke_sampler_run():
    x, y, z = sym.symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    Q, _offset = qubo.Compile(expr).get_qubo()
    api_key = os.environ.get("TYTAN_API_KEY", "foobar")
    sampler = ZekeSampler()
    result = sampler.run(qubo=Q, shots=1, api_key=api_key)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] is not None
    assert result[0][0]["y"] is not None
    assert result[0][0]["z"] is not None
    assert result[0][1] is not None
    assert result[0][2] is not None

import sympy as sym
import pytest
from tests.support.custom_stub import make_vcr_request
from tytan import qubo
from tytan.sampler import NQSLocalSampler
import vcr.stubs.httpx_stubs


@pytest.mark.vcr(
    match_on=["uri", "method"],
    custom_patches=(
        (vcr.stubs.httpx_stubs, "_make_vcr_request", make_vcr_request),
    ),
)
def test_nqs_local_sampler_run():
    x, y, z = sym.symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    Q, _offset = qubo.Compile(expr).get_qubo()
    sampler = NQSLocalSampler()
    result = sampler.run(qubo=Q)
    assert result is not None
    assert result["result"] is not None
    assert result["result"]["x"] == 0
    assert result["result"]["y"] == 0
    assert result["result"]["z"] == 0
    assert result["energy"] == 0
    assert result["time"] is not None

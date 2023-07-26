import pytest
from tests.support.custom_stub import make_vcr_request
from tytan import symbols, Compile
from tytan.sampler import NQSLocalSampler
import vcr.stubs.httpx_stubs




@pytest.mark.vcr(
    match_on=["uri", "method"],
    custom_patches=(
        (vcr.stubs.httpx_stubs, "_make_vcr_request", make_vcr_request),
    ),
)
def test_nqs_local_sampler_run():
    x, y, z = symbols("x y z")
    expr = 3 * x**2 + 2 * x * y + 4 * y**2 + z**2 + 2 * x * z + 2 * y * z
    qubo, offset = Compile(expr).get_qubo()
    sampler = NQSLocalSampler()
    result = sampler.run(qubo)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] == 0
    assert result[0][0]["y"] == 0
    assert result[0][0]["z"] == 0
    assert result[0][1] == 0 #energy
    assert result[0][2] is not None #occ
    assert result[0][3] is not None #time

from tytan import symbols, Compile
from tytan.sampler import SASampler


def test_sa_sampler_run():
    x, y, z = symbols("x y z")
    expr = (x + y + z - 2)**2
    qubo, offset = Compile(expr).get_qubo()
    sampler = SASampler()
    result = sampler.run(qubo)
    for r in result:
        print(r)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] is not None
    assert result[0][0]["y"] is not None
    assert result[0][0]["z"] is not None
    assert result[0][1] is not None
    assert result[0][2] is not None

def test_sa_sampler_run_with_seed():
    x, y, z = symbols("x y z")
    expr = (x + y + z - 2)**2
    qubo, offset = Compile(expr).get_qubo()
    
    #1
    print('try 1, ', end='')
    sampler = SASampler(seed=0)
    result = sampler.run(qubo)
    print(result[0][0]["x"], result[0][0]["y"], result[0][0]["z"], result[0][2])
    x = result[0][0]["x"]
    y = result[0][0]["y"]
    z = result[0][0]["z"]
    count = result[0][2]
    
    #2-
    for i in range(2, 10):
        print(f'try {i}, ', end='')
        sampler = SASampler(seed=0)
        result = sampler.run(qubo)
        print(result[0][0]["x"], result[0][0]["y"], result[0][0]["z"], result[0][2])
        assert result[0][0]["x"] == x
        assert result[0][0]["y"] == y
        assert result[0][0]["z"] == z
        assert result[0][2] == count

import pytest
import numpy as np
from tytan import symbols_list, symbols_define


# Convert symbol array to string array (for comparison)
def get_symbol_names(symbol_array):
    return np.vectorize(lambda x: str(x))(symbol_array)


def test_symbols_list_dim1():
    expected_names = np.array(["q0", "q1"])

    result = symbols_list((2), "q{}")
    result_names = get_symbol_names(result)

    assert np.array_equal(result_names, expected_names)


def test_symbols_list_dim2():
    expected_names = np.array([["q0_0", "q0_1", "q0_2"], ["q1_0", "q1_1", "q1_2"]])

    result = symbols_list((2, 3), "q{}_{}")
    result_names = get_symbol_names(result)

    assert np.array_equal(result_names, expected_names)


def test_symbols_list_dim3():
    expected_names = np.array(
        [
            [
                ["q0_0_0", "q0_0_1", "q0_0_2", "q0_0_3"],
                ["q0_1_0", "q0_1_1", "q0_1_2", "q0_1_3"],
                ["q0_2_0", "q0_2_1", "q0_2_2", "q0_2_3"],
            ],
            [
                ["q1_0_0", "q1_0_1", "q1_0_2", "q1_0_3"],
                ["q1_1_0", "q1_1_1", "q1_1_2", "q1_1_3"],
                ["q1_2_0", "q1_2_1", "q1_2_2", "q1_2_3"],
            ],
        ]
    )

    result = symbols_list((2, 3, 4), "q{}_{}_{}")
    result_names = get_symbol_names(result)

    assert np.array_equal(result_names, expected_names)


def test_symbols_define_dim1():
    expected = "q0 = symbols('q0')\r\n" + "q1 = symbols('q1')"

    result = symbols_define((2), "q{}")

    assert result == expected, f"Expected:\n{expected}\nBut got:\n{result}"


def test_symbols_define_dim2():
    expected = (
        "q0_0 = symbols('q0_0')\r\n"
        + "q0_1 = symbols('q0_1')\r\n"
        + "q0_2 = symbols('q0_2')\r\n"
        + "q1_0 = symbols('q1_0')\r\n"
        + "q1_1 = symbols('q1_1')\r\n"
        + "q1_2 = symbols('q1_2')"
    )

    result = symbols_define((2, 3), "q{}_{}")

    assert result == expected, f"Expected:\n{expected}\nBut got:\n{result}"


def test_symbols_define_dim3():
    expected = (
        "q0_0_0 = symbols('q0_0_0')\r\n"
        + "q0_0_1 = symbols('q0_0_1')\r\n"
        + "q0_0_2 = symbols('q0_0_2')\r\n"
        + "q0_0_3 = symbols('q0_0_3')\r\n"
        + "q0_1_0 = symbols('q0_1_0')\r\n"
        + "q0_1_1 = symbols('q0_1_1')\r\n"
        + "q0_1_2 = symbols('q0_1_2')\r\n"
        + "q0_1_3 = symbols('q0_1_3')\r\n"
        + "q0_2_0 = symbols('q0_2_0')\r\n"
        + "q0_2_1 = symbols('q0_2_1')\r\n"
        + "q0_2_2 = symbols('q0_2_2')\r\n"
        + "q0_2_3 = symbols('q0_2_3')\r\n"
        + "q1_0_0 = symbols('q1_0_0')\r\n"
        + "q1_0_1 = symbols('q1_0_1')\r\n"
        + "q1_0_2 = symbols('q1_0_2')\r\n"
        + "q1_0_3 = symbols('q1_0_3')\r\n"
        + "q1_1_0 = symbols('q1_1_0')\r\n"
        + "q1_1_1 = symbols('q1_1_1')\r\n"
        + "q1_1_2 = symbols('q1_1_2')\r\n"
        + "q1_1_3 = symbols('q1_1_3')\r\n"
        + "q1_2_0 = symbols('q1_2_0')\r\n"
        + "q1_2_1 = symbols('q1_2_1')\r\n"
        + "q1_2_2 = symbols('q1_2_2')\r\n"
        + "q1_2_3 = symbols('q1_2_3')"
    )

    result = symbols_define((2, 3, 4), "q{}_{}_{}")

    assert result == expected, f"Expected:\n{expected}\nBut got:\n{result}"

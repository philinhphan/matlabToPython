import pytest
from example_function import add_numbers


def test_add_numbers():
    assert add_numbers(5, 10) == 15
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    assert add_numbers(3.5, 2.5) == 6
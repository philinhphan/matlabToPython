import pytest
from example_script import add_numbers

def test_add_numbers_positive_integers():
    assert add_numbers(5, 10) == 15


def test_add_numbers_negative_integers():
    assert add_numbers(-5, -10) == -15


def test_add_numbers_zero():
    assert add_numbers(0, 0) == 0
    assert add_numbers(5, 0) == 5
    assert add_numbers(0, 5) == 5


def test_add_numbers_floats():
    assert add_numbers(5.5, 4.5) == 10.0


def test_add_numbers_edge_cases():
    assert add_numbers(float('inf'), 10) == float('inf')
    assert add_numbers(-float('inf'), -10) == -float('inf')
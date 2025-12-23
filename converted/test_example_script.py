import pytest
from example_script import add_numbers

def test_add_numbers_positive():
    assert add_numbers(5, 10) == 15


def test_add_numbers_negative():
    assert add_numbers(-5, -10) == -15


def test_add_numbers_zero():
    assert add_numbers(0, 0) == 0


def test_add_numbers_edge_case():
    assert add_numbers(1.5, 2.5) == 4.0
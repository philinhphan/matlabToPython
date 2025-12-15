import pytest
from example_script import add_numbers


def test_add_numbers_with_positive_integers():
    assert add_numbers(5, 10) == 15


def test_add_numbers_with_negative_integers():
    assert add_numbers(-1, -2) == -3


def test_add_numbers_with_mixed_sign_integers():
    assert add_numbers(-1, 1) == 0


def test_add_numbers_with_zero():
    assert add_numbers(0, 0) == 0
    assert add_numbers(0, 5) == 5
    assert add_numbers(5, 0) == 5


def test_add_numbers_with_floats():
    assert add_numbers(0.1, 0.2) == pytest.approx(0.3)
    assert add_numbers(-0.1, 0.1) == pytest.approx(0.0)


def test_add_numbers_with_large_numbers():
    assert add_numbers(1e10, 1e10) == 2e10


def test_add_numbers_with_special_cases():
    # Edge case with large negative and positive numbers
    assert add_numbers(-1e10, 1e10) == 0.0
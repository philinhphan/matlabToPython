import pytest
from example_function import add_numbers
import numpy as np
import matplotlib.pyplot as plt


# Test the add_numbers function using test cases
@pytest.mark.parametrize("a, b, expected", [
    (5, 10, 15),
    (-1, 1, 0),
    (0, 0, 0),
    (3.5, 2.5, 6)
])
def test_add_numbers(a, b, expected):
    assert add_numbers(a, b) == expected


def test_main_plot():
    data = np.arange(1, 11)
    squared = data ** 2

    # Generate the plot
    plt.figure()
    plt.plot(data, squared, 'b-', linewidth=2)
    plt.grid(True)

    # Get the current axes
    ax = plt.gca()
    line = ax.lines[0]

    # Check that the x-data and y-data match expected values
    np.testing.assert_array_equal(line.get_xdata(), data)
    np.testing.assert_array_equal(line.get_ydata(), squared)

    # Ensure grid is enabled
    assert ax.xaxis.get_major_ticks()[0].gridOn
    assert ax.yaxis.get_major_ticks()[0].gridOn
import numpy as np
import matplotlib.pyplot as plt

def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

# Variables
x = 5
y = 10

# Call the add_numbers function
sum_result = add_numbers(x, y)

print(f'The sum of {x} and {y} is {sum_result}')

# Create a simple plot
data = np.arange(1, 11)
squared = data ** 2

plt.figure()
plt.plot(data, squared, 'b-', linewidth=2)
plt.title('Squared Numbers')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
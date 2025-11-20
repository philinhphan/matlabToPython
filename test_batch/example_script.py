import matplotlib.pyplot as plt
from example_function import add_numbers

# Example script that uses add_numbers function

def main():
    x = 5
    y = 10

    # Call the add_numbers function
    sum_result = add_numbers(x, y)

    print(f'The sum of {x} and {y} is {sum_result}')

    # Create a simple plot
    data = range(1, 11)
    squared = [d ** 2 for d in data]

    plt.figure()
    plt.plot(data, squared, 'b-', linewidth=2)
    plt.title('Squared Numbers')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
% Example script that uses add_numbers function
clc; clear;

x = 5;
y = 10;

% Call the add_numbers function
sum_result = add_numbers(x, y);

fprintf('The sum of %d and %d is %d\n', x, y, sum_result);

% Create a simple plot
data = 1:10;
squared = data.^2;

figure;
plot(data, squared, 'b-', 'LineWidth', 2);
title('Squared Numbers');
xlabel('Input');
ylabel('Output');
grid on;

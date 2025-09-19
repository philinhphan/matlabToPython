% calculate_sum.m
% A simple function to demonstrate Matlab to Python conversion.

function s = calculate_sum(n)
    % This function calculates the sum of the first n integers.
    s = 0;
    for i = 1:n
        s = s + i;
    end
    fprintf('The sum of the first %d integers is: %d\n', n, s);
end
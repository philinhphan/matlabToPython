%% Complex MATLAB Script for Conversion Testing
% Author: ChatGPT
% Date: 2025-10-22
% Description:
% This script combines multiple MATLAB functionalities:
% - Signal processing (filtering, FFT, wavelets)
% - Linear algebra and numerical methods
% - Image processing
% - Optimization and data fitting
% - Structures, cells, and OOP usage
% Used to test conversion tools between MATLAB and Python.

clc; clear; close all;

%% SECTION 1: Synthetic Signal Generation and Analysis
fs = 1000;              % Sampling frequency
t = 0:1/fs:1-1/fs;      % Time vector
f1 = 50; f2 = 120;      % Frequencies
signal = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t + pi/4) + 0.2*randn(size(t));

% Apply FFT
N = length(signal);
freq = (0:N-1)*(fs/N);
spectrum = abs(fft(signal));

% Plot
figure('Name','Signal Analysis');
subplot(2,1,1); plot(t, signal); title('Original Signal'); xlabel('Time (s)');
subplot(2,1,2); plot(freq(1:N/2), spectrum(1:N/2)); title('FFT Magnitude'); xlabel('Frequency (Hz)');

%% SECTION 2: Filter Design and Application
lpFilt = designfilt('lowpassfir','PassbandFrequency',100,'StopbandFrequency',150,...
    'PassbandRipple',0.5,'StopbandAttenuation',60,'DesignMethod','equiripple','SampleRate',fs);
filteredSignal = filter(lpFilt, signal);

figure('Name','Filtered Signal');
plot(t,filteredSignal,'r'); hold on; plot(t,signal,'b--');
legend('Filtered','Original'); title('Low-pass Filtering');

%% SECTION 3: Numerical Computation – Eigenvalues and Differential Equations
A = rand(5); 
[eigVec,eigVal] = eig(A);

% Solve a differential equation: dy/dt = -2y + sin(t)
odeFunc = @(t,y) -2*y + sin(t);
[tSol, ySol] = ode45(odeFunc, [0 10], 1);

figure('Name','ODE Solution');
plot(tSol,ySol); title('dy/dt = -2y + sin(t)');

%% SECTION 4: Image Processing
img = peaks(256);
noisyImg = img + 0.1*randn(size(img));
filteredImg = imgaussfilt(noisyImg,2);

figure('Name','Image Processing');
subplot(1,3,1); imagesc(img); title('Original Image'); axis image off;
subplot(1,3,2); imagesc(noisyImg); title('Noisy Image'); axis image off;
subplot(1,3,3); imagesc(filteredImg); title('Gaussian Filtered'); axis image off;
colormap jet;

%% SECTION 5: Optimization Example (Nonlinear Fit)
xData = linspace(0,4*pi,100);
yTrue = 3*sin(2*xData) + 2;
yData = yTrue + 0.5*randn(size(yTrue));

modelFun = @(b,x) b(1)*sin(b(2)*x) + b(3);
beta0 = [2,1,1]; % Initial guess
optFunc = @(b) sum((yData - modelFun(b,xData)).^2);
betaEst = fminsearch(optFunc, beta0);

figure('Name','Curve Fitting');
plot(xData, yData, 'ko'); hold on;
plot(xData, modelFun(betaEst,xData), 'r-', 'LineWidth', 2);
title(sprintf('Fitted: y = %.2f sin(%.2fx) + %.2f', betaEst));
legend('Data','Fit');

%% SECTION 6: Cell Arrays, Structs, and Tables
dataCell = {'Signal', signal; 'Filtered', filteredSignal; 'Spectrum', spectrum};
dataStruct = struct('A',A,'Eigenvalues',diag(eigVal),'BetaEst',betaEst);
dataTable = table(xData', yData', modelFun(betaEst,xData)', ...
    'VariableNames', {'x','yMeasured','yFitted'});

disp('Data Structure Example:');
disp(dataStruct);
disp(head(dataTable));

%% SECTION 7: Wavelet Transform and Reconstruction
[c,l] = wavedec(signal,4,'db4');
approx = appcoef(c,l,'db4');
reconstructed = waverec(c,l,'db4');

figure('Name','Wavelet Reconstruction');
plot(t,signal,'b'); hold on;
plot(t,reconstructed,'r--');
title('Wavelet Reconstruction');
legend('Original','Reconstructed');

%% SECTION 8: Object-Oriented Programming Example
classdef SimpleFilter
    properties
        Cutoff
        Fs
    end
    methods
        function obj = SimpleFilter(cutoff, fs)
            obj.Cutoff = cutoff;
            obj.Fs = fs;
        end
        function y = apply(obj, x)
            [b,a] = butter(4, obj.Cutoff/(obj.Fs/2));
            y = filtfilt(b,a,x);
        end
    end
end

%% SECTION 9: Use Custom Class
sf = SimpleFilter(0.2*fs, fs);
try
    filteredY = sf.apply(signal);
    figure('Name','OOP Filtered Signal');
    plot(t,filteredY);
    title('Filtered Signal (via OOP Class)');
catch ME
    warning('OOP section skipped: %s', ME.message);
end

%% SECTION 10: Parallel Computation
if license('test','Distrib_Computing_Toolbox')
    parfor i = 1:10
        A = rand(500);
        eig(A);
    end
else
    disp('Parallel Computing Toolbox not available; skipping parfor test.');
end

%% End of Script
disp('Complex MATLAB test script completed successfully.');

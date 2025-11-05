import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, firwin, lfilter
from scipy.fft import fft
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from scipy.io.matlab import loadmat
import pywt

# Function Definitions
class SimpleFilter:
    def __init__(self, cutoff, fs):
        self.cutoff = cutoff
        self.fs = fs

    def apply(self, x):
        b, a = butter(4, self.cutoff / (self.fs / 2))
        return filtfilt(b, a, x)

# SECTION 1: Synthetic Signal Generation and Analysis
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1 / fs)  # Time vector
f1, f2 = 50, 120  # Frequencies
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t + np.pi / 4) + 0.2 * np.random.randn(*t.shape)

# Apply FFT
N = len(signal)
freq = np.arange(N) * (fs / N)
spectrum = np.abs(fft(signal))

# Plot
plt.figure('Signal Analysis')
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.subplot(2, 1, 2)
plt.plot(freq[:N // 2], spectrum[:N // 2])
plt.title('FFT Magnitude')
plt.xlabel('Frequency (Hz)')
plt.show()

# SECTION 2: Filter Design and Application
b = firwin(numtaps=101, cutoff=100, fs=fs)
filteredSignal = lfilter(b, 1, signal)

plt.figure('Filtered Signal')
plt.plot(t, filteredSignal, 'r', label='Filtered')
plt.plot(t, signal, 'b--', label='Original')
plt.legend()
plt.title('Low-pass Filtering')
plt.show()

# SECTION 3: Numerical Computation – Eigenvalues and Differential Equations
A = np.random.rand(5, 5)
eigVal, eigVec = np.linalg.eig(A)

# Solve a differential equation: dy/dt = -2y + sin(t)
def ode_func(t, y):
    return -2 * y + np.sin(t)

sol = solve_ivp(ode_func, [0, 10], [1], t_eval=np.linspace(0, 10, 100))

plt.figure('ODE Solution')
plt.plot(sol.t, sol.y[0])
plt.title('dy/dt = -2y + sin(t)')
plt.show()

# SECTION 4: Image Processing
img = loadmat('peaks.mat')['peaks'] # Load the peak image or simulate a similar one
noisyImg = img + 0.1 * np.random.randn(*img.shape)
filteredImg = gaussian_filter(noisyImg, 2)

plt.figure('Image Processing', figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='jet')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(noisyImg, cmap='jet')
plt.title('Noisy Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(filteredImg, cmap='jet')
plt.title('Gaussian Filtered')
plt.axis('off')
plt.show()

# SECTION 5: Optimization Example (Nonlinear Fit)
xData = np.linspace(0, 4 * np.pi, 100)
yTrue = 3 * np.sin(2 * xData) + 2
yData = yTrue + 0.5 * np.random.randn(len(yTrue))

model_fun = lambda b, x: b[0] * np.sin(b[1] * x) + b[2]

# Minimize the sum of squares error
opt_func = lambda b: np.sum((yData - model_fun(b, xData))**2)

result = minimize(opt_func, [2, 1, 1])
beta_est = result.x

plt.figure('Curve Fitting')
plt.plot(xData, yData, 'ko', label='Data')
plt.plot(xData, model_fun(beta_est, xData), 'r-', label='Fit', linewidth=2)
plt.title(f'Fitted: y = {beta_est[0]:.2f} sin({beta_est[1]:.2f}x) + {beta_est[2]:.2f}')
plt.legend()
plt.show()

# SECTION 6: Data Structures
# Data stored in Python lists and dictionaries
dataCell = {'Signal': signal, 'Filtered': filteredSignal, 'Spectrum': spectrum}
dataStruct = {'A': A, 'Eigenvalues': eigVal, 'BetaEst': beta_est}
dataTable = {'x': xData, 'yMeasured': yData, 'yFitted': model_fun(beta_est, xData)}

print('Data Struct Example:')
print(dataStruct)
print('Data Table Example:')
print(dataTable)

# SECTION 7: Wavelet Transform and Reconstruction
coeffs = pywt.wavedec(signal, 'db4', level=4)

# For approximation reconstruction, we need to use coeffs directly
decomposed = pywt.waverec(coeffs, 'db4')  # Reconstructed full signal

plt.figure('Wavelet Reconstruction')
plt.plot(t, signal, 'b', label='Original')
plt.plot(t, decomposed, 'r--', label='Reconstructed')
plt.title('Wavelet Reconstruction')
plt.legend()
plt.show()

# SECTION 8-9: Object-Oriented Programming Example
sf = SimpleFilter(0.2 * fs, fs)
filteredY = sf.apply(signal)

plt.figure('OOP Filtered Signal')
plt.plot(t, filteredY)
plt.title('Filtered Signal (via OOP Class)')
plt.show()

# SECTION 10: Parallel Computation
print('Checking for parallel processing capabilities...')
try:
    from joblib import Parallel, delayed
    
    def compute_eigenvalues(size):
        A = np.random.rand(size, size)
        return np.linalg.eig(A)
    
    results = Parallel(n_jobs=4)(delayed(compute_eigenvalues)(500) for _ in range(10))
    print('Parallel computation was successful.')
except ImportError:
    print('Parallel Computing Toolbox not available; skipping parallel computation test.')

print('Complex Python script completed successfully.')
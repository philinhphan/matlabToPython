import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import firwin, filtfilt, butter, sosfiltfilt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import pywt

# SECTION 1: Synthetic Signal Generation and Analysis
fs = 1000      # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
f1, f2 = 50, 120
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t + np.pi/4) + 0.2 * np.random.randn(len(t))

# Apply FFT
N = len(signal)
freq = np.fft.fftfreq(N, 1/fs)
spectrum = np.abs(fft(signal))

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.subplot(2, 1, 2)
plt.plot(freq[:N//2], spectrum[:N//2])
plt.title('FFT Magnitude')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# SECTION 2: Filter Design and Application
numtaps = 150  # Filter order
cutoff = 100.0
# Filter coefficients
b = firwin(numtaps, cutoff, window='hamming', fs=fs)
filtered_signal = filtfilt(b, [1.0], signal)

plt.figure(figsize=(10, 4))
plt.plot(t, filtered_signal, 'r', label='Filtered')
plt.plot(t, signal, 'b--', alpha=0.7, label='Original')
plt.legend()
plt.title('Low-pass Filtering')
plt.show()

# SECTION 3: Numerical Computation – Eigenvalues and Differential Equations
A = np.random.rand(5, 5)
eig_vals, eig_vecs = np.linalg.eig(A)

# Solve a differential equation: dy/dt = -2y + sin(t)
def ode_func(y, t):
    return -2 * y + np.sin(t)

time_span = np.arange(0, 10, 0.01)
y0 = 1.0

y_sol = odeint(ode_func, y0, time_span)

plt.figure(figsize=(10, 4))
plt.plot(time_span, y_sol)
plt.title('dy/dt = -2y + sin(t)')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.show()

# SECTION 4: Image Processing
img = np.random.rand(256, 256)
noisy_img = img + 0.1 * np.random.randn(256, 256)
filtered_img = ndimage.gaussian_filter(noisy_img, sigma=2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='jet')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap='jet')
plt.title('Noisy Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(filtered_img, cmap='jet')
plt.title('Gaussian Filtered')
plt.axis('off')
plt.show()

# SECTION 5: Optimization Example (Nonlinear Fit)
x_data = np.linspace(0, 4 * np.pi, 100)
y_true = 3 * np.sin(2 * x_data) + 2
y_data = y_true + 0.5 * np.random.randn(*y_true.shape)

# Model function
def model_func(beta, x):
    return beta[0] * np.sin(beta[1] * x) + beta[2]

# Objective function
def objective_func(beta):
    return np.sum((y_data - model_func(beta, x_data)) ** 2)

beta0 = [2, 1, 1]  # Initial guess
result = minimize(objective_func, beta0)
beta_est = result.x

plt.figure()
plt.plot(x_data, y_data, 'ko', label='Data')
plt.plot(x_data, model_func(beta_est, x_data), 'r-', linewidth=2, label='Fit')
plt.title(f'Fitted: y = {beta_est[0]:.2f} sin({beta_est[1]:.2f}x) + {beta_est[2]:.2f}')
plt.legend()
plt.show()

# SECTION 6: Cell Arrays, Structs, and Tables
import pandas as pd
data_dict = {
    "A": A,
    "Eigenvalues": eig_vals,
    "BetaEst": beta_est
}
data_table = pd.DataFrame({
    'x': x_data.flatten(),
    'yMeasured': y_data.flatten(),
    'yFitted': model_func(beta_est, x_data).flatten()
})

print("Data Structure Example:")
print(data_dict)
print(data_table.head())

# SECTION 7: Wavelet Transform and Reconstruction
coeffs = pywt.wavedec(signal, 'db4', level=4)
reconstructed = pywt.waverec(coeffs, 'db4')

plt.figure(figsize=(10, 4))
plt.plot(t, signal, 'b', label='Original')
plt.plot(t[:len(reconstructed)], reconstructed, 'r--', label='Reconstructed')
plt.title('Wavelet Reconstruction')
plt.legend()
plt.show()

# SECTION 8 & 9: Object-Oriented Programming (Class and Usage)
class SimpleFilter:
    def __init__(self, cutoff, fs):
        self.cutoff = cutoff
        self.fs = fs

    def apply(self, x):
        b, a = butter(4, self.cutoff / (self.fs / 2), btype='low')
        return filtfilt(b, a, x)

sf = SimpleFilter(0.2 * fs, fs)

try:
    filtered_y = sf.apply(signal)
    plt.figure()
    plt.plot(t, filtered_y)
    plt.title('Filtered Signal (via OOP Class)')
    plt.show()
except Exception as e:
    print(f'OOP section skipped: {e}')

# SECTION 10: Parallel Computation
import multiprocessing as mp
def compute_eig(_):
    A = np.random.rand(500, 500)
    np.linalg.eig(A)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(compute_eig, range(10))

print('Complex MATLAB test script completed successfully.')


'''
Emergenz, 
'''
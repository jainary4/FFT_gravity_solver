import numpy as np  
import matplotlib.pyplot as plt  
  
# Load FFT output  
fft_data = np.loadtxt('fft_output.txt', skiprows=2)  
indices = fft_data[:, 0]  
fft_real = fft_data[:, 1]  
fft_imag = fft_data[:, 2]  
fft_magnitude = np.sqrt(fft_real**2 + fft_imag**2)  
  
# Load IFFT output  
ifft_data = np.loadtxt('ifft_output.txt', skiprows=2)  
ifft_real = ifft_data[:, 1]  
  
# Plot  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  
  
# FFT magnitude spectrum  
ax1.plot(indices, fft_magnitude)  
ax1.set_title('FFT Magnitude Spectrum')  
ax1.set_xlabel('Frequency Bin')  
ax1.set_ylabel('Magnitude')  
  
# Reconstructed time-domain signal  
ax2.plot(indices, ifft_real)  
ax2.set_title('Inverse FFT (Reconstructed Signal)')  
ax2.set_xlabel('Sample Index')  
ax2.set_ylabel('Amplitude')  
  
plt.tight_layout()  
plt.savefig('fft_visualization.png')  
plt.show()
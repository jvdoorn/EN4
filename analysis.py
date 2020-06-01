import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from common import balance, file_name, view, correction

# Read the audio file
rate, audio = wavfile.read(file_name)
audio = balance(audio)

count = audio.shape[0]  # Number of data points
length = count / rate  # Length of the recording (seconds)

print(f'Audio length: {length:.2f}s')
print(f'Audio rate: {rate}; Count: {count}')

transform = np.abs(np.fft.fft(audio))  # Apply Fourier
time_series = np.linspace(0, rate, count)  # Create the corresponding time series

print(f'Maximum magnitude: {np.amax(transform[:count // 2]) / count:.0f}')

# Prepare matplotlib
plt.plot(time_series[:count // 2] * correction, transform[:count // 2] / count)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

if view is None or view.__len__() != 2:
    sub_title = 'Rate: {rate} [$1/s$]'
    save_file = f'figs/{file_name[5:-4]}.png'
    plt.xlim(20, 20000)  # Audible frequency range
else:
    sub_title = f'Rate: {rate} [$1/s$]; Zoomed ({view[0]} - {view[1]} Hz)'
    save_file = f'figs/{file_name[5:-4]}-zoomed-{view[0]}-{view[1]}.png'
    plt.xlim(view[0], view[1])

plt.title(f'Fourier analysis of {file_name}\n{sub_title} (c: {correction:.3f})')
plt.savefig(save_file)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

from common import file_name, view

# Read the audio file
count = 157632
rate = 48000
time_series = np.linspace(0, count/rate, count)  # Create the corresponding time series
audio = 50 * np.sin(2 * np.pi * time_series)

transform = np.fft.fft(audio)  # Apply Fourier

# Prepare matplotlib
plt.plot(time_series[:count // 2], np.abs(transform)[:count // 2] * 1 / count)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

if view is None or view.__len__() != 2:
    sub_title = 'Rate: {rate} [$1/s$]'
    plt.xlim(20, 20000)  # Audible frequency range
else:
    sub_title = f'Rate: {rate} [$1/s$]; Zoomed ({view[0]} - {view[1]} Hz)'
    plt.xlim(view[0], view[1])

plt.title(f'Fourier analysis of {file_name}\n{sub_title}')

plt.savefig(f'figs/{file_name[5:-4]}.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from skimage import util

from common import balance, file_name, correction

window_size = 1024  # Window size (must be a multiple of 2 (performance))

# Read the audio file
rate, audio = wavfile.read(file_name)
audio = balance(audio)

count = audio.shape[0]  # Number of data points
length = count / rate  # Length of the recording (seconds)

print(f'Audio length: {length:.2f}s')

# Prepare matplotlib
figure, axes = plt.subplots(nrows=2, sharex='all')

axes[0].set_title(f'Audio visualization of {file_name}')
axes[1].set_title(f'Spectogram of {file_name}')

axes[0].plot(np.arange(count) / rate, audio)
axes[0].set_ylabel('Magnitude')

# Slice the audio
slices = util.view_as_windows(audio, window_shape=(window_size,), step=100)
print(f'Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}')

win = np.hanning(window_size + 1)[:-1]
slices = slices * win

slices = slices.T
print('Shape of slices:', slices.shape)

# Apply Fourier
spectrum = np.fft.fft(slices, axis=0)[:window_size // 2 + 1:-1]
spectrum = np.abs(spectrum)
spectrum = 20 * np.log10(spectrum / np.max(spectrum))

axes[1].imshow(spectrum, origin='lower', cmap='inferno', extent=(0, length, 0, rate / 2))
axes[1].axis('tight')
axes[1].set_ylabel('Frequency [Hz]')
axes[1].set_xlabel('Time [s]')

figure.savefig(f'figs/{file_name[5:-4]}-visual.png')
figure.show()

from numpy import mean
from numpy.core.multiarray import ndarray

file_name = "data/neartest.wav"  # File to analyse
view = (0, 2500)  # Optional view to zoom in on specific frequency range
correction = 1.031  # Corrects for errors in the analysis and devices


def balance(signal: ndarray) -> ndarray:
    """
    Attempts to balance the sound signal from L/R.

    :param signal: the audio signal of a WAV file.
    :return: the balanced signal or the original signal.
    """
    try:
        return mean(signal, axis=1)
    except IndexError:
        return signal

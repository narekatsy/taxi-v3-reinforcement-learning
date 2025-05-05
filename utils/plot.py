import numpy as np

def moving_average(data, window_size=100):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

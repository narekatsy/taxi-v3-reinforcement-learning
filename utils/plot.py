import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=100):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_rewards(rewards, label=None, window_size=100):
    """
    Plots the smoothed rewards over episodes.

    Args:
        rewards: List of episode rewards.
        label: Optional label for the plot (e.g., hyperparameter config).
        window_size: Window size for smoothing the curve.
    """
    smoothed = moving_average(rewards, window_size)
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title("SARSA Training Performance")
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
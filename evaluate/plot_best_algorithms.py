import os
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import moving_average

# Map each algorithm to its best config file name
best_configs = {
    "Q-Learning": "results/q_learning/α0.1_γ0.95_ε-decay0.99.npy",
    "SARSA": "results/sarsa/α0.1_γ0.99_ε-decay0.995.npy",
    "Expected SARSA": "results/expected_sarsa/α0.2_γ0.99_ε-decay0.996.npy",
    "Monte Carlo": "results/monte_carlo/α0.3_γ0.98_ε-decay0.998.npy",
    "n-Step SARSA": "results/n_step_sarsa/α0.1_γ0.95_ε-decay0.995_n1.npy",
    "Dyna-Q": "results/dyna_q/α0.25_γ0.96_ε-decay0.995_plan3.npy",
    "Dyna-Q+": "results/dyna_q_plus/α0.25_γ0.96_ε-decay0.995_plan3_κ0.002.npy"
}

plt.figure(figsize=(12, 6))

for label, path in best_configs.items():
    if os.path.exists(path):
        rewards = np.load(path)
        smoothed = moving_average(rewards, window_size=500)
        plt.plot(smoothed, label=label)
    else:
        print(f"[Warning] Missing: {path}")

plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward")
plt.title("Comparison of Best Configs Across Algorithms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/best_algorithms_comparison.png")
plt.show()

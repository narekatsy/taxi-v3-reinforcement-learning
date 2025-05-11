import os
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import moving_average

def compute_metrics(rewards, last_n=500):
    return np.mean(rewards[-last_n:])

def evaluate_all(results):
    summary = []
    for label, rewards in results.items():
        avg_final = compute_metrics(rewards)
        summary.append((label, avg_final))

    summary.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Dyna-Q+ Config Performance Ranking ===")
    for i, (label, avg) in enumerate(summary, 1):
        print(f"{i}. {label}: Avg Final Reward = {avg:.2f}")
    return summary

def plot_all_curves(results, output_path="results/dyna_q_plus/all_configs_comparison.png"):
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        smoothed = moving_average(rewards)
        plt.plot(smoothed, label=label)
    plt.title("Dyna-Q+: All Configs Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.legend()
    plt.grid()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def main():
    results_dir = "results/dyna_q_plus"
    rewards = {}
    for file in os.listdir(results_dir):
        if file.endswith(".npy"):
            label = file.replace(".npy", "").replace("_", " ")
            rewards[label] = np.load(os.path.join(results_dir, file))

    evaluate_all(rewards)
    plot_all_curves(rewards)

if __name__ == "__main__":
    main()

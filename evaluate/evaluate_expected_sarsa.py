import os
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import moving_average

def compute_metrics(rewards, last_n=500):
    avg_final = np.mean(rewards[-last_n:])
    return avg_final

def evaluate_all(results):
    summary = []
    for label, rewards in results.items():
        avg_final = compute_metrics(rewards)
        summary.append((label, avg_final))

    summary.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Expected SARSA Config Performance Ranking ===")
    for rank, (label, avg_reward) in enumerate(summary, 1):
        print(f"{rank}. {label}: Final Avg Reward = {avg_reward:.2f}")

    return summary

def plot_all_curves(results, output_path="results/expected_sarsa/all_configs_comparison.png"):
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        smoothed = moving_average(rewards)
        plt.plot(smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Expected SARSA: All Configs Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def main():
    results_dir = "results/expected_sarsa"
    rewards = {}

    for file in os.listdir(results_dir):
        if file.endswith(".npy"):
            label = file.replace(".npy", "").replace("_", " ")
            rewards[label] = np.load(os.path.join(results_dir, file))

    evaluate_all(rewards)
    plot_all_curves(rewards)

if __name__ == "__main__":
    main()

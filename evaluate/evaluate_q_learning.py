import numpy as np
import matplotlib.pyplot as plt
from utils.plot import moving_average
import os

def load_results(results_dir):
    rewards_data = {}
    for file in os.listdir(results_dir):
        if file.endswith(".npy"):
            label = file.replace(".npy", "").replace("_", " ")
            rewards = np.load(os.path.join(results_dir, file))
            rewards_data[label] = rewards
    return rewards_data

def compute_metrics(rewards, last_n=500):
    avg_final = np.mean(rewards[-last_n:])
    return avg_final

def evaluate_all(results):
    summary = []

    for label, rewards in results.items():
        avg_final = compute_metrics(rewards)
        summary.append((label, avg_final))

    # Sort by performance
    summary.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Q-Learning Config Performance Ranking ===")
    for rank, (label, avg_reward) in enumerate(summary, 1):
        print(f"{rank}. {label}: Final Avg Reward = {avg_reward:.2f}")

    return summary

def plot_all_curves(results):
    plt.figure(figsize=(12, 6))

    for label, rewards in results.items():
        smoothed = moving_average(rewards)
        plt.plot(smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Q-Learning: All Configs Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/q_learning/all_configs_comparison.png")
    plt.show()

def main():
    results_dir = "results/q_learning"
    rewards = {}

    for file in os.listdir(results_dir):
        if file.endswith(".png"):
            continue  # Skip images
        if file.endswith(".npy"):
            label = file.replace(".npy", "").replace("_", " ")
            rewards[label] = np.load(os.path.join(results_dir, file))

    # Evaluation
    evaluate_all(rewards)
    plot_all_curves(rewards)

if __name__ == "__main__":
    main()

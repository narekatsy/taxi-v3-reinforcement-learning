import gymnasium as gym
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from agents.expected_sarsa import expected_sarsa
from utils.plot import moving_average

def load_hyperparameters(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_plot(rewards, label, output_dir):
    smoothed = moving_average(rewards, window_size=100)
    plt.figure(figsize=(8, 4))
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title(f"Expected SARSA: {label}")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"expected_sarsa_{label.replace(' ', '_').replace('=', '').replace(',', '')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main(args):
    env = gym.make("Taxi-v3")
    configs = load_hyperparameters(args.config)
    results = {}

    for cfg in configs:
        alpha = cfg["alpha"]
        gamma = cfg["gamma"]
        epsilon_decay = cfg["epsilon_decay"]
        label = f"α={alpha}, γ={gamma}, ε-decay={epsilon_decay}"

        print(f"Training Expected SARSA with: {label}")
        q_table, rewards = expected_sarsa(
            env=env,
            num_episodes=args.episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            epsilon_decay=epsilon_decay,
            min_epsilon=0.01,
            verbose=args.verbose
        )

        results[label] = rewards
        filename = f"{label.replace(' ', '_').replace('=', '').replace(',', '')}.npy"
        os.makedirs(args.output, exist_ok=True)
        np.save(os.path.join(args.output, filename), rewards)
        save_plot(rewards, label, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Expected SARSA with multiple hyperparameter configs.")
    parser.add_argument("--config", type=str, default="configs/expected_sarsa.json", help="Path to config JSON file")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--output", type=str, default="results/expected_sarsa", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()
    main(args)

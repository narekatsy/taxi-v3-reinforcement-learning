import gymnasium as gym
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from agents.sarsa import sarsa
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
    plt.title(f"SARSA: {label}")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    filename = label.replace(" ", "_").replace("=", "").replace(",", "")
    plt.savefig(os.path.join(output_dir, f"sarsa_{filename}.png"))
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

        print(f"Training SARSA with: {label}")
        q_table, rewards = sarsa(
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
        filename = label.replace(" ", "_").replace("=", "").replace(",", "")
        np.save(os.path.join(args.output, f"{filename}.npy"), rewards)
        save_plot(rewards, label, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARSA with multiple hyperparameter configs.")
    parser.add_argument("--config", type=str, default="configs/sarsa.json", help="Path to config JSON file")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--output", type=str, default="results/sarsa", help="Output directory for plots")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()

    main(args)

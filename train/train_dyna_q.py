import gymnasium as gym
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from agents.dyna_q import dyna_q
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
    plt.title(f"Dyna-Q: {label}")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dyna_q_{label.replace(' ', '_').replace('=', '').replace(',', '')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def main(args):
    env = gym.make("Taxi-v3")
    configs = load_hyperparameters(args.config)

    for cfg in configs:
        alpha = cfg["alpha"]
        gamma = cfg["gamma"]
        epsilon_decay = cfg["epsilon_decay"]
        planning_steps = cfg["planning_steps"]
        label = f"α={alpha}, γ={gamma}, ε-decay={epsilon_decay}, plan={planning_steps}"

        print(f"Training Dyna-Q with: {label}")
        q_table, rewards = dyna_q(
            env=env,
            num_episodes=args.episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            planning_steps=planning_steps,
            epsilon_decay=epsilon_decay,
            min_epsilon=0.01,
            verbose=args.verbose
        )

        filename = f"{label.replace(' ', '_').replace('=', '').replace(',', '')}.npy"
        os.makedirs(args.output, exist_ok=True)
        np.save(os.path.join(args.output, filename), rewards)
        save_plot(rewards, label, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dyna-Q with multiple hyperparameter configs.")
    parser.add_argument("--config", type=str, default="configs/dyna_q.json")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--output", type=str, default="results/dyna_q")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)

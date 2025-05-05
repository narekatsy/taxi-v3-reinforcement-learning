import argparse
import subprocess
import sys

def run_training(algo, episodes):
    if algo == "q_learning":
        subprocess.run([
            sys.executable, "-m", "train.train_q_learning",
            "--config", "configs/q_learning.json",
            "--episodes", str(episodes),
            "--verbose"
        ])
    elif algo == "sarsa":
        subprocess.run([
            sys.executable, "-m", "train.train_sarsa",
            "--config", "configs/sarsa.json",
            "--episodes", str(episodes),
            "--verbose"
        ])
    else:
        print(f"Unknown algorithm: {algo}")

def run_evaluation(algo):
    if algo == "q_learning":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_q_learning"])
    elif algo == "sarsa":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_sarsa"])
    else:
        print(f"Unknown algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description="Main runner for RL project")
    parser.add_argument("--algo", choices=["q_learning", "sarsa"], required=True, help="Algorithm to run")
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="all", help="Run mode")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes (for training)")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        run_training(args.algo, args.episodes)
    
    if args.mode in ("evaluate", "all"):
        run_evaluation(args.algo)

if __name__ == "__main__":
    main()

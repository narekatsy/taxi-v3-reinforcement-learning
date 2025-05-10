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
    elif algo == "expected_sarsa":
        subprocess.run([
            sys.executable, "-m", "train.train_expected_sarsa",
            "--config", "configs/expected_sarsa.json",
            "--episodes", str(episodes),
            "--verbose"
    ])
    elif algo == "monte_carlo":
        subprocess.run([
            sys.executable, "-m", "train.train_monte_carlo",
            "--config", "configs/monte_carlo.json",
            "--episodes", str(episodes),
            "--verbose"
    ])
    elif algo == "n_step_sarsa":
        subprocess.run([
            sys.executable, "-m", "train.train_n_step_sarsa",
            "--config", "configs/n_step_sarsa.json",
            "--episodes", str(episodes),
            "--verbose"
    ])
    elif algo == "dyna_q":
        subprocess.run([
            sys.executable, "-m", "train.train_dyna_q",
            "--config", "configs/dyna_q.json",
            "--episodes", str(episodes),
            "--verbose"
    ])
    elif algo == "dyna_q_plus":
        subprocess.run([
            sys.executable, "-m", "train.train_dyna_q_plus",
            "--config", "configs/dyna_q_plus.json",
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
    elif algo == "expected_sarsa":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_expected_sarsa"])
    elif algo == "monte_carlo":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_monte_carlo"])
    elif algo == "n_step_sarsa":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_n_step_sarsa"])
    elif algo == "dyna_q":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_dyna_q"])
    elif algo == "dyna_q_plus":
        subprocess.run([sys.executable, "-m", "evaluate.evaluate_dyna_q_plus"])
    else:
        print(f"Unknown algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description="Main runner for RL project")
    parser.add_argument("--algo", choices=["q_learning", "sarsa", "expected_sarsa", "monte_carlo", "n_step_sarsa", "dyna_q", "dyna_q_plus"],
                        required=True, help="Algorithm to run")
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="all", help="Run mode")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes (for training)")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        run_training(args.algo, args.episodes)
    
    if args.mode in ("evaluate", "all"):
        run_evaluation(args.algo)

if __name__ == "__main__":
    main()

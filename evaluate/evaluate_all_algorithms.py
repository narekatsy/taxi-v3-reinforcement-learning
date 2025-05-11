import os
import numpy as np

RESULTS_BASE_DIR = "results"
LAST_N_EPISODES = 600

def compute_avg_final_reward(rewards, last_n=LAST_N_EPISODES):
    return np.mean(rewards[-last_n:]) if len(rewards) >= last_n else np.mean(rewards)

def evaluate_algorithm(algorithm_dir):
    algorithm_name = os.path.basename(algorithm_dir)
    summary = []

    for file in os.listdir(algorithm_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(algorithm_dir, file)
            rewards = np.load(file_path)
            avg = compute_avg_final_reward(rewards)
            cumulative = np.sum(rewards)
            label = file.replace(".npy", "").replace("_", " ")
            summary.append((label, avg, cumulative))

    summary.sort(key=lambda x: x[1], reverse=True)  # or x[2] for cumulative
    return algorithm_name, summary


def evaluate_all_algorithms():
    best_configs = {}
    output_path = "results/evaluation_summary.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n🔍 Performance Evaluation by Algorithm:\n")
    for algorithm in os.listdir(RESULTS_BASE_DIR):
        path = os.path.join(RESULTS_BASE_DIR, algorithm)
        if not os.path.isdir(path):
            continue

        algo_name, summary = evaluate_algorithm(path)
        print(f"📌 {algo_name.upper()}:")
        for rank, (label, avg, cumulative) in enumerate(summary[:5], 1):
            print(f"  {rank}. {label} — Avg Reward: {avg:.2f}, Cumulative: {cumulative:.0f}")
        
        if summary:
            best_configs[algo_name] = summary[0]  # Best config for the algorithm
        print()

    print("🏆 Best Configs Comparison Across Algorithms:\n")
    sorted_best = sorted(best_configs.items(), key=lambda x: x[1][1], reverse=True)
    for rank, (algo, (label, avg, cumulative)) in enumerate(sorted_best, 1):
        print(f"{rank}. {algo}: {label} — Avg Reward: {avg:.2f}, Cumulative: {cumulative:.0f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Best Configurations Per Algorithm ===\n")
        for algo, (label, avg, cumulative) in best_configs.items():
            f.write(f"{algo}: {label} — Avg Reward: {avg:.2f}, Cumulative: {cumulative:.0f}\n")
        
        f.write("\n=== Best Algorithms Overall ===\n")
        for rank, (algo, (label, avg, cumulative)) in enumerate(sorted_best, 1):
            f.write(f"{rank}. {algo}: {label} — Avg Reward: {avg:.2f}, Cumulative: {cumulative:.0f}\n")


if __name__ == "__main__":
    evaluate_all_algorithms()

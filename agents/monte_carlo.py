import numpy as np
import random
from collections import defaultdict

def monte_carlo_control(env, num_episodes, alpha, gamma, epsilon, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_data = []
        total_reward = 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, _ = env.step(action)
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

        G = 0
        visited_state_actions = set()
        for t in reversed(range(len(episode_data))):
            state_t, action_t, reward_t = episode_data[t]
            G = gamma * G + reward_t
            if (state_t, action_t) not in visited_state_actions:
                visited_state_actions.add((state_t, action_t))
                returns_sum[(state_t, action_t)] += G
                returns_count[(state_t, action_t)] += 1
                q_table[state_t][action_t] = returns_sum[(state_t, action_t)] / returns_count[(state_t, action_t)]

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward}")

    return q_table, episode_rewards

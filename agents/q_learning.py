import numpy as np
import random

def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    """
    Q-learning algorithm implementation.

    Args:
        env: OpenAI Gym environment
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Decay rate per episode
        min_epsilon: Minimum exploration rate
        verbose: If True, prints progress every 100 episodes

    Returns:
        q_table: Learned Q-values
        episode_rewards: List of total rewards per episode
    """
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"[Q-Learning] Episode {episode + 1}/{num_episodes} | Reward: {total_reward}")

    return q_table, episode_rewards

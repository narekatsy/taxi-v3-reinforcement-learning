import numpy as np
import random

def sarsa(env, num_episodes, alpha, gamma, epsilon, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])
        done = False
        total_reward = 0

        while not done:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            next_action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[next_state])

            # SARSA update
            q_table[state, action] += alpha * (
                reward + gamma * q_table[next_state, next_action] - q_table[state, action]
            )

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

    return q_table, episode_rewards

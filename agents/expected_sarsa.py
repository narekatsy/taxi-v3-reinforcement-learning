import numpy as np
import random

def expected_sarsa(env, num_episodes, alpha, gamma, epsilon, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Compute expected Q
            action_probs = np.ones(env.action_space.n) * (epsilon / env.action_space.n)
            best_action = np.argmax(q_table[next_state])
            action_probs[best_action] += (1.0 - epsilon)

            expected_q = np.dot(q_table[next_state], action_probs)
            q_table[state, action] += alpha * (reward + gamma * expected_q - q_table[state, action])

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward}")

    return q_table, episode_rewards

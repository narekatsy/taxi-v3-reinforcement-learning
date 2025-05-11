import numpy as np
import random
from collections import defaultdict

def dyna_q(env, num_episodes, alpha, gamma, epsilon, planning_steps, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    model = defaultdict(list)
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Update Q-table
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            # Update model
            model[(state, action)] = (next_state, reward)

            # Planning
            for _ in range(planning_steps):
                (s, a), (s_next, r) = random.choice(list(model.items()))
                q_table[s, a] += alpha * (r + gamma * np.max(q_table[s_next]) - q_table[s, a])

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} | Total reward: {total_reward}")

    return q_table, episode_rewards

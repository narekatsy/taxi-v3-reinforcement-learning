import numpy as np
import random
from collections import defaultdict

def dyna_q_plus(env, num_episodes, alpha, gamma, epsilon, planning_steps, kappa, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    model = dict()
    last_visited = defaultdict(lambda: 0)
    episode_rewards = []
    time_step = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            time_step += 1

            # ε-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Update Q-table
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            # Update model and last visited time
            model[(state, action)] = (next_state, reward)
            last_visited[(state, action)] = time_step

            # Planning with exploration bonus
            for _ in range(planning_steps):
                (s, a) = random.choice(list(model.keys()))
                s_next, r = model[(s, a)]
                tau = time_step - last_visited[(s, a)]
                bonus = kappa * np.sqrt(tau)
                q_table[s, a] += alpha * ((r + bonus) + gamma * np.max(q_table[s_next]) - q_table[s, a])

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} | Total reward: {total_reward}")

    return q_table, episode_rewards

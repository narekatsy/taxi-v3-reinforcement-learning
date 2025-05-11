import numpy as np
import random

def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon, n, epsilon_decay=0.99, min_epsilon=0.01, verbose=False):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = select_action(q_table, state, epsilon, env.action_space.n)
        states, actions, rewards = [state], [action], [0]

        T = float("inf")
        t = 0
        total_reward = 0

        while True:
            if t < T:
                next_state, reward, done, truncated, _ = env.step(actions[t])
                total_reward += reward
                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1
                    actions.append(None)
                else:
                    next_action = select_action(q_table, next_state, epsilon, env.action_space.n)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += gamma**n * q_table[states[tau + n], actions[tau + n]]

                s_tau = states[tau]
                a_tau = actions[tau]
                q_table[s_tau, a_tau] += alpha * (G - q_table[s_tau, a_tau])

            if tau == T - 1:
                break

            t += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward}")

    return q_table, episode_rewards

def select_action(q_table, state, epsilon, action_space_size):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_space_size - 1)
    return np.argmax(q_table[state])

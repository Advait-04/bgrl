from data import get_dataset
import numpy as np
import matplotlib.pyplot as plt

def train_model(env, agent):
    data = get_dataset(300)
    dopamine_values, acetyl_values = data
    common_length = len(dopamine_values)

    num_episodes = 300
    batch_size = 32
    cumulative_rewards = []
    convergence_threshold = 1900
    convergence_rate = None
    sample_efficiency_threshold = 1900
    sample_efficiency = None
    avg_reward = []

    for episode in range(num_episodes):
        state_episode = []
        total_reward = 0

        dopamine_value = dopamine_values[episode % common_length]
        acetyl_value = acetyl_values[episode % common_length]

        state = env.reset()
        state_episode.append(state)

        while True:
            state_index = env.states.index(state)
            action = agent.select_action(state_index)

            while env.actions[action] not in env.transition_probabilities[state]:
                action = agent.select_action(state_index)

            next_state, reward, done, _ = env.step(env.actions[action], dopamine_value, acetyl_value)
            next_state_index = env.states.index(next_state)
            agent.memory.append((state_index, action, reward, next_state_index, done))

            agent.update_q_network(batch_size)

            total_reward += reward
            state = next_state

            if done:
                break

        if episode % 10 == 0:
            agent.update_target_network()

        cumulative_rewards.append(total_reward)

        if len(cumulative_rewards) >= 10:
            current_avg_reward = np.mean(cumulative_rewards[-10:])
            avg_reward.append(current_avg_reward)
            if convergence_rate is None and current_avg_reward >= convergence_threshold:
                convergence_rate = episode

        if sample_efficiency is None and total_reward >= sample_efficiency_threshold:
            sample_efficiency = episode + 1

        agent.exploration_prob = max(agent.epsilon_min, agent.epsilon_decay * agent.exploration_prob)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.exploration_prob}")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward, label='DQN')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Convergence Rate')
    plt.show()

    if convergence_rate is not None:
        print(f"Convergence achieved at episode {convergence_rate} with average reward {convergence_threshold}.")
    else:
        print("Convergence threshold not reached.")

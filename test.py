from data import get_dataset
import numpy as np
import matplotlib.pyplot as plt

def test_model(env, agent):
    data = get_dataset(10)
    dopamine_values, acetyl_values = data

    no_iterations = 50
    direct_array = ["Cortex", "Striatum", "GPi", "Thalamus"]
    indirect_array = ["Cortex", "Striatum", "GPe", "STN", "GPi", "Thalamus"]
    hyperdirect_array = ["Cortex", "STN", "GPi", "Thalamus"]
    accuracy_array = []
    accuracy_per_10_episodes = []

    for i in range(no_iterations):
        state = env.reset()
        done = False
        output_array = [state]

        dopamine_value = dopamine_values[i % len(dopamine_values)]
        acetyl_value = acetyl_values[i % len(acetyl_values)]

        agent.exploration_prob = 0.2
        print(i, " Finally chosen pathway: ")
        while not done:
            state_index = env.states.index(state)
            action = agent.select_action(state_index)

            while env.actions[action] not in env.transition_probabilities[state]:
                action = agent.select_action(state_index)

            next_state, reward, done, _ = env.step(env.actions[action], dopamine_value, acetyl_value)

            print(f"Transition: {env.states[state_index]} -> {next_state}, Reward: {reward}")

            output_array.append(next_state)
            state = next_state

        if dopamine_value < 39.6:
            accuracy_array.append(output_array == indirect_array)
        elif 39.6 < dopamine_value < 59.6:
            accuracy_array.append(output_array == hyperdirect_array)
        elif 39.6 < dopamine_value < 195.8:
            accuracy_array.append(output_array == direct_array)

        if (i + 1) % 10 == 0:
            accuracy = accuracy_array.count(True) / len(accuracy_array)
            accuracy_per_10_episodes.append(accuracy)
            accuracy_array = []  # Reset accuracy array for the next 10 episodes

    # Plot the accuracy graph for every 10 episodes
    plt.plot(range(10, no_iterations + 1, 10), accuracy_per_10_episodes, label='Accuracy per 10 episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Episodes')
    plt.show()

    print("Accuracy: ", accuracy_per_10_episodes)

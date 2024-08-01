import tensorflow as tf
import numpy as np
from collections import deque


class DQNAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.001, discount_factor=0.99, exploration_prob=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Build Q-network
        self.q_network = self.build_q_network()

        # Target Q-network (for stability)
        self.target_q_network = self.build_q_network()
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Experience replay buffer
        self.memory = deque(maxlen=2000)

    def build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(
                self.state_space_size,), dtype=tf.float32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space_size)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.action_space_size)
        else:
            state_one_hot = np.zeros(self.state_space_size)
            state_one_hot[state] = 1
            state_one_hot = state_one_hot.reshape(1, -1)
            q_values = self.q_network.predict(state_one_hot)
            return np.argmax(q_values[0])

    def update_q_network(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.memory[i] for i in batch])

        states = np.eye(self.state_space_size)[np.array(states)]
        next_states = np.eye(self.state_space_size)[np.array(next_states)]

        q_values = self.q_network.predict(states)
        next_q_values = self.target_q_network.predict(next_states)

        for i in range(batch_size):
            target = rewards[i] + self.discount_factor * \
                np.max(next_q_values[i]) * (1 - dones[i])
            q_values[i, actions[i]] = target

        self.q_network.fit(states, q_values, verbose=0)

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

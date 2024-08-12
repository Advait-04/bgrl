from enum import CONTINUOUS
import gym
import numpy as np


class BasalGangliaMDP(gym.Env):
    def __init__(self):
        super().__init__()

        self.states = ['Cortex', 'Striatum', 'GPe', 'STN', 'GPi', 'Thalamus']

        self.actions = ["activation", "inhibition"]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.transition_probabilities = {
            "Cortex": {
                "activation": {"Striatum": 0.5, "STN": 0.5},
                "inhibition": {"Striatum": 1.0}
            },
            "Striatum": {
                "inhibition": {'GPe': 0.5, 'GPi': 0.5},
            },
            "GPe": {
                "inhibition": {"STN": 1.0},
            },
            "STN": {
                "activation": {"GPi": 1.0},
            },
            "GPi": {
                "inhibition": {"Thalamus": 1.0},
            },
            "Thalamus": {
                "activation": {"Cortex": 1.0},
            }
        }

        self.rewards = {
            ("Cortex", "activation", "Striatum"): 1.0,
            ("Cortex", "inhibition", "Striatum"): 1.0,

            ("Striatum", "inhibition", "GPe"): 1.0,
            ("Striatum", "inhibition", "GPi"): 1.0,

            ("GPe", "inhibition", "STN"): 1.0,

            ("STN", "activation", "GPi"): 1.0,

            ("GPi", "inhibition", "Thalamus"): 1.0,

            ("Thalamus", "activation", "Cortex"): 1.0
        }

        self.state = "Cortex"

    def calculate_rewards(self, state, action, next_state, dopamine, acetyl, levodopa):
        direct_pathway = [("Cortex", "activation", "Striatum"),
                          ("Striatum", "inhibition", "GPe"),
                          ("GPe", "inhibition", "STN"),
                          ("STN", "activation", "GPi"),
                          ("GPi", "inhibition", "Thalamus")]

        hyper_direct_pathway = [("Cortex", "activation", "STN"),
                                ("STN", "activation", "GPi"),
                                ("GPi", "inhibition", "Thalamus")]

        indirect_pathway = [("Cortex", "activation", "Striatum"),
                            ("Striatum", "inhibition", "GPi"),
                            ("GPi", "inhibition", "Thalamus")]

        if dopamine < 39.6 and acetyl > 2.5:
            if (state, action, next_state) in direct_pathway:
                return self.rewards.get((state, action, next_state), 1.0)
            else:
                return -1.0  # Negative reward for incorrect path

        elif 39.6 <= dopamine <= 59.6 and 1 <= acetyl <= 2:
            if (state, action, next_state) in hyper_direct_pathway:
                return self.rewards.get((state, action, next_state), 1.0)
            else:
                return -1.0  # Negative reward for incorrect path

        elif 59.6 < dopamine <= 195.8 and acetyl < 0.5:
            if (state, action, next_state) in indirect_pathway:
                return self.rewards.get((state, action, next_state), 1.0)
            else:
                return -1.0  # Negative reward for incorrect path

        else:
            return 0.0  # Neutral reward if none of the conditions match

    def step(self, action, dopamine, acetyl, levodopa):
        next_state_probs = self.transition_probabilities[self.state][action]
        next_state = np.random.choice(
            list(next_state_probs.keys()), p=list(next_state_probs.values()))

        # reward = self.rewards.get((self.state, action, next_state), 0)

        reward = self.calculate_rewards(
            self.state, action, next_state, dopamine, acetyl, levodopa)

        self.state = next_state

        done = next_state == 'Thalamus'  # Terminal state
        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.state = "Cortex"
        return self.state

    def render(self):
        print(self.state)

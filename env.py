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
                "activation": {"Striatum": 1.0},
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
          if(dopamine <= 39.6 and acetyl > 2.5 and levodopa < 100):
            if(state == "Cortex" and next_state == "Striatum"):
              if(action == "activation"):
                return self.rewards.get((self.state, action, next_state), 0) * -100
              elif(action == "inhibition"):
                return self.rewards.get((self.state, action, next_state), 0) * 250

            elif(state=="Striatum" and next_state=="GPe"):
                return self.rewards.get((self.state, action, next_state), 0) * 400

            elif(state=="Striatum" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * -250

            elif(state=="GPe" and next_state=="STN"):
                return self.rewards.get((self.state, action, next_state), 0) * 500

            elif(state=="STN" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * 600

            elif(state=="GPi" and next_state=="Thalamus"):
                return self.rewards.get((self.state, action, next_state), 0) * 800

            else:
                raise Exception("Get good bro")


          elif(dopamine <= 39.6 and acetyl > 2.5 and levodopa >= 100 and levodopa <=250):
            if(state == "Cortex" and next_state == "Striatum"):
              if(action == "activation"):
                return self.rewards.get((self.state, action, next_state), 0) * 200

              elif(action == "inhibition"):
                return self.rewards.get((self.state, action, next_state), 0) * -150

            elif(state=="Striatum" and next_state=="GPe"):
                return self.rewards.get((self.state, action, next_state), 0) * -200

            elif(state=="Striatum" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * 450

            elif(state=="GPe" and next_state=="STN"):
                return self.rewards.get((self.state, action, next_state), 0) * -300

            elif(state=="STN" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * -350

            elif(state=="GPi" and next_state=="Thalamus"):
                return self.rewards.get((self.state, action, next_state), 0) * 1000

            else:
                raise Exception("Get good bro")


          elif(dopamine > 39.6 and dopamine <= 195.8 and acetyl >=0.5 and acetyl <= 2.5):
            if(state == "Cortex" and next_state == "Striatum"):
              if(action == "activation"):
                return self.rewards.get((self.state, action, next_state), 0) * 200

              elif(action == "inhibition"):
                return self.rewards.get((self.state, action, next_state), 0) * -150

            elif(state=="Striatum" and next_state=="GPe"):
                return self.rewards.get((self.state, action, next_state), 0) * -200

            elif(state=="Striatum" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * 550

            elif(state=="GPe" and next_state=="STN"):
                return self.rewards.get((self.state, action, next_state), 0) * -300

            elif(state=="STN" and next_state=="GPi"):
                return self.rewards.get((self.state, action, next_state), 0) * -350

            elif(state=="GPi" and next_state=="Thalamus"):
                return self.rewards.get((self.state, action, next_state), 0) * 1000

            else:
                raise Exception("Get good bro")

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

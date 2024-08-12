from data import get_dataset
from env import BasalGangliaMDP
from agent import DQNAgent

if __name__=="__main__":
    env = BasalGangliaMDP()
    agent = DQNAgent(state_space_size=len(env.states), action_space_size=env.action_space.n)

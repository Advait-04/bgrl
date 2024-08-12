from data import get_dataset
from env import BasalGangliaMDP
from agent import DQNAgent
from test import test_model
from train import train_model

if __name__=="__main__":
    print("hello world")
    env = BasalGangliaMDP()
    agent = DQNAgent(state_space_size=len(env.states), action_space_size=env.action_space.n)

    train_model(env, agent)
    test_model(env, agent)

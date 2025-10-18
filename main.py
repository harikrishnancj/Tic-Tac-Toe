from train import train
from play import play_vs_agent


env, agent = train(episodes=20000, print_every=2000)

play_vs_agent(env, agent)

from env import TicTacToe
from agent import QLearningAgent
import random

def train(episodes=50000, print_every=5000):
    env = TicTacToe()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False
        player = 1

        while not done:
            available_moves = env.available_moves()
            action = agent.choose_action(state, available_moves)

            next_state, reward, done = env.step(action, player)

            if not done:
                opp_moves = env.available_moves()
                if opp_moves:
                    opp_action = random.choice(opp_moves)
                    next_state, opp_reward, done = env.step(opp_action, -1)
                    if done:
                        reward = -1 * opp_reward

            agent.learn(state, action, reward, next_state, done, env.available_moves())
            state = next_state

        if (episode + 1) % print_every == 0:
            print(f"Episode {episode+1}/{episodes} — epsilon={agent.epsilon:.3f}")

    print("Training complete ✅")
    return env, agent

if __name__ == "__main__":
    train()

from env import TicTacToe
import numpy as np

def play_vs_agent(env, agent):
    state = env.reset()
    done = False
    print("You are O. Agent is X.")
    env.render()

    while not done:
        # Agent move
        action = np.argmax(agent.get_qs(state))
        state, _, done = env.step(action, 1)
        print("\nAgent move:")
        env.render()
        if done:
            break

        # Human move with validation
        while True:
            try:
                human_move = int(input("Enter your move (0–8): "))
                if human_move not in env.available_moves():
                    print("❌ Invalid move! Cell occupied or out of range. Try again.")
                    continue
                state, _, done = env.step(human_move, -1)
                break
            except ValueError:
                print("⚠️ Enter a valid number (0-8).")

        print("\nYour move:")
        env.render()

    if env.winner == 1:
        print("Agent wins! 😎")
    elif env.winner == -1:
        print("You win! 🎉")
    else:
        print("It's a draw.")

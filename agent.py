import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]

    def heuristic_move(self, board, player):
        """Check for immediate win or block opponent."""
        b = np.array(board).reshape(3, 3)
        # Check agent win
        for i in range(9):
            row, col = divmod(i, 3)
            if b[row, col] != 0:
                continue
            b[row, col] = player
            if self.check_win(b, player):
                return i
            b[row, col] = 0
        # Check opponent block
        opponent = -player
        for i in range(9):
            row, col = divmod(i, 3)
            if b[row, col] != 0:
                continue
            b[row, col] = opponent
            if self.check_win(b, opponent):
                return i
            b[row, col] = 0
        return None

    def check_win(self, board, player):
        for i in range(3):
            if abs(sum(board[i, :])) == 3 or abs(sum(board[:, i])) == 3:
                return True
        if abs(sum(np.diag(board))) == 3 or abs(sum(np.diag(np.fliplr(board)))) == 3:
            return True
        return False

    def choose_action(self, state, available_moves):
        # Try heuristic first
        move = self.heuristic_move(state, 1)
        if move is not None:
            return move

        # Else use epsilon-greedy Q-learning
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        qs = self.get_qs(state)
        valid_qs = [(i, qs[i]) for i in available_moves]
        return max(valid_qs, key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state, done, available_moves):
        qs = self.get_qs(state)
        next_qs = self.get_qs(next_state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max([next_qs[i] for i in available_moves])
        qs[action] = qs[action] + self.alpha * (target - qs[action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

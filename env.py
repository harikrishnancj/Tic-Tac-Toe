import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0=empty, 1=X, -1=O
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def available_moves(self):
        return [i for i, x in enumerate(self.board.flatten()) if x == 0]

    def step(self, move, player):
        if self.done or self.board.flatten()[move] != 0:
            return self.get_state(), -10, True  # illegal move penalty

        row, col = divmod(move, 3)
        self.board[row, col] = player

        reward, done, winner = self.check_status(player)
        self.done = done
        self.winner = winner
        return self.get_state(), reward, done

    def check_status(self, player):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return (1 if player == 1 else -1), True, player
        if abs(sum(np.diag(self.board))) == 3 or abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return (1 if player == 1 else -1), True, player
        if not 0 in self.board:
            return 0.5, True, 0
        return 0, False, None

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\n".join(["|".join(symbols[x] for x in row) for row in self.board]))
        print("-" * 5)

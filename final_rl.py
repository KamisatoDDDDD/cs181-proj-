import numpy as np
import random
from environment import Game
from expectimax import simulate_move


# =========================
# Feature（自动适配版）
# =========================
def extract_features(board):
    size = board.shape[0]

    empty = np.sum(board == 0)

    log_board = np.log2(board + 1)

    smooth = -(
        np.sum(np.abs(np.diff(log_board, axis=0))) +
        np.sum(np.abs(np.diff(log_board, axis=1)))
    )

    merge = 0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] == board[r, c + 1] and board[r, c] > 0:
                merge += 1

    # ===== stable monotonic pattern（自动适配 size）=====
    idx = np.arange(size * size).reshape(size, size)
    weights = np.exp(-idx / (size * size))

    monotonic = np.sum(log_board * weights)

    return np.array([empty, smooth, merge, monotonic], dtype=np.float32)


# =========================
# Agent
# =========================
class FastRL:
    def __init__(self):
        self.w = np.array([35, -1.0, 2.0, 1.5], dtype=np.float32)

    def value(self, board):
        return np.dot(self.w, extract_features(board))

    def update(self, phi, target, value, lr=0.0015):
        td = target - value
        td = np.clip(td, -3, 3)
        self.w += lr * td * phi
        self.w = np.clip(self.w, -80, 80)


# =========================
# policy (stable greedy + ε)
# =========================
def select_action(game, agent, epsilon=0.1):

    valid = []
    for a in range(4):
        nb, moved, _ = simulate_move(game.board, a)
        if moved:
            valid.append(a)

    if not valid:
        return None

    # exploration
    if random.random() < epsilon:
        return random.choice(valid)

    # exploitation
    best_a, best_v = None, -1e9

    for a in valid:
        nb, _, reward = simulate_move(game.board, a)
        v = agent.value(nb)
        score = reward + v

        if score > best_v:
            best_v = score
            best_a = a

    return best_a


# =========================
# training
# =========================
def train_fast(episodes=3000):

    agent = FastRL()
    gamma = 0.98

    for ep in range(episodes):

        game = Game()
        state = game.reset()

        while True:

            action = select_action(game, agent, epsilon=0.12)

            if action is None:
                break

            next_state, reward, done, _ = game.step(action)

            phi = extract_features(state)
            v = agent.value(state)

            if done:
                target = reward
            else:
                target = reward + gamma * agent.value(next_state)

            agent.update(phi, target, v)

            state = next_state

            if done:
                break

        if ep % 200 == 0:
            print(f"[RL] ep={ep}, score={game.score}, w={agent.w}")

    return agent


# =========================
# main
# =========================
if __name__ == "__main__":
    agent = train_fast(3000)
    np.save("model.npy", agent.w)  

    print("\nFINAL WEIGHTS:")
    print(agent.w)
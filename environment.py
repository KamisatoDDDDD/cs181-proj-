import numpy as np
import random

class Game:
    def __init__(self, size=5):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.step_count = 0
        self.score = 0

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.step_count = 0
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def step(self, action):
        # action: 0: up, 1: down, 2: left, 3: right
        moved, merge_score = self._move(action)
        if not moved:
            return self.get_state(), -1, False, {'valid': False}
        self.score += merge_score
        self.step_count += 1
        self._add_random_tile()
        if self.step_count % 8 == 0:
            self._exam_accident()
        if self.step_count % 12 == 0:
            self._coach_guidance()
        done = self._is_game_over()
        return self.get_state(), merge_score, done, {'valid': True}

    def _move(self, action):
        moved = False
        score = 0
        board = self.board.copy()

        # 方向变换
        if action == 0:      # 上
            board = board.T
        elif action == 1:    # 下
            board = np.flip(board.T, axis=1)
        elif action == 3:    # 右
            board = np.flip(board, axis=1)
        # action == 2 (左) 不变

        for i in range(self.size):
            row = board[i]
            compressed = row[row != 0]
            new_row = []
            j = 0
            while j < len(compressed):
                if j + 1 < len(compressed) and compressed[j] == compressed[j+1]:
                    merged_val = compressed[j] * 2
                    new_row.append(merged_val)
                    score += merged_val
                    j += 2
                else:
                    new_row.append(compressed[j])
                    j += 1
            # 补零到 self.size
            new_row = np.array(new_row + [0] * (self.size - len(new_row)), dtype=np.int32)
            if not np.array_equal(row, new_row):
                moved = True
            board[i] = new_row

        # 还原方向
        if action == 0:
            board = board.T
        elif action == 1:
            board = np.flip(board, axis=1).T
        elif action == 3:
            board = np.flip(board, axis=1)

        self.board = board
        return moved, score

    def _add_random_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r, c] = 2 if random.random() < 0.9 else 4

    def _exam_accident(self):
        # 只降级等级在 [省三(4), 国三(64)] 范围内的方块，国二(128)及以上豁免
        candidates = [(r, c) for r in range(self.size) for c in range(self.size)
                    if 4 <= self.board[r, c] < 128]
        if not candidates:
            return
        r, c = random.choice(candidates)
        self.board[r, c] = max(2, self.board[r, c] // 2)  # 最低降到省四(2)

    def _coach_guidance(self):
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r, c] = 16 if random.random() < 0.9 else 32

    def _is_game_over(self):
        if np.any(self.board == 0):
            return False
        # 水平相邻检查
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] == self.board[r, c+1]:
                    return False
        # 垂直相邻检查
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r, c] == self.board[r+1, c]:
                    return False
        return True
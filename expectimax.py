import numpy as np
import random

# ==================== 评估函数权重（针对合成国家队调整） ====================
WEIGHTS = {
    'empty': 35.0,
    'monotonicity': -1.2,
    'smoothness': -1.5,
    'corner_max': 12.0,
}

TILE_PROBS = [(2, 0.9), (4, 0.1)]
COACH_TILE_PROBS = [(16, 0.9), (32, 0.1)]

# 采样设置（兼顾速度与精度）
SAMPLE_THRESHOLD = 6
SAMPLE_SIZE = 8
ACCIDENT_SAMPLE_MAX = 5
COACH_SAMPLE_MAX = 5

# ==================== 辅助函数 ====================

def is_game_over(board):
    size = board.shape[0]
    if np.any(board == 0):
        return False
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] == board[r, c+1]:
                return False
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] == board[r+1, c]:
                return False
    return True

def simulate_move(board, action):
    size = board.shape[0]
    temp_board = board.copy()
    moved = False
    score = 0
    if action == 0:
        temp_board = temp_board.T
    elif action == 1:
        temp_board = np.flip(temp_board.T, axis=1)
    elif action == 3:
        temp_board = np.flip(temp_board, axis=1)

    for i in range(size):
        row = temp_board[i]
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
        new_row = np.array(new_row + [0] * (size - len(new_row)), dtype=np.int32)
        if not np.array_equal(row, new_row):
            moved = True
        temp_board[i] = new_row

    if action == 0:
        temp_board = temp_board.T
    elif action == 1:
        temp_board = np.flip(temp_board, axis=1).T
    elif action == 3:
        temp_board = np.flip(temp_board, axis=1)

    return temp_board, moved, score

def _monotonicity_score(arr):
    score = 0
    for i in range(len(arr)-1):
        if arr[i] >= arr[i+1]:
            score += 1
        else:
            score -= 1
    return score

def evaluate(board):
    size = board.shape[0]
    log_board = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])
    score = 0.0

    # 空格数
    empty_count = np.sum(board == 0)
    score += WEIGHTS['empty'] * empty_count

    # 单调性
    mono_total = 0
    for r in range(size):
        row = log_board[r]
        mono_total += max(_monotonicity_score(row), _monotonicity_score(row[::-1]))
    for c in range(size):
        col = log_board[:, c]
        mono_total += max(_monotonicity_score(col), _monotonicity_score(col[::-1]))
    score += WEIGHTS['monotonicity'] * mono_total

    # 平滑度
    smooth_total = 0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] > 0 and board[r, c+1] > 0:
                smooth_total -= abs(log_board[r, c] - log_board[r, c+1])
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] > 0 and board[r+1, c] > 0:
                smooth_total -= abs(log_board[r, c] - log_board[r+1, c])
    score += WEIGHTS['smoothness'] * smooth_total

    # 角落最大值
    corners = [board[0, 0], board[0, size-1], board[size-1, 0], board[size-1, size-1]]
    max_corner = max(corners)
    max_val = np.max(board)
    if max_corner == max_val:
        score += WEIGHTS['corner_max'] * max_val
    else:
        score -= WEIGHTS['corner_max'] * max_val * 0.5

    return score

# ==================== 置换表 ====================
trans_table = {}

def tt_key(board, step_mod):
    return (tuple(board.flatten()), step_mod)

# ==================== 搜索函数 ====================

def chance_node(board, step_count, depth):
    size = board.shape[0]
    mod = step_count % 24
    key = tt_key(board, mod)
    if key in trans_table:
        return trans_table[key]

    if depth == 0 or is_game_over(board):
        val = evaluate(board)
        trans_table[key] = val
        return val

    # 1. 正常随机生成方块
    empty_cells = list(zip(*np.where(board == 0)))
    states = []
    probs = []
    if empty_cells:
        if len(empty_cells) <= SAMPLE_THRESHOLD:
            for cell in empty_cells:
                for tile_val, prob in TILE_PROBS:
                    nb = board.copy()
                    nb[cell] = tile_val
                    states.append(nb)
                    probs.append((1.0 / len(empty_cells)) * prob)
        else:
            sampled = random.sample(empty_cells, min(SAMPLE_SIZE, len(empty_cells)))
            for cell in sampled:
                for tile_val, prob in TILE_PROBS:
                    nb = board.copy()
                    nb[cell] = tile_val
                    states.append(nb)
                    probs.append((1.0 / len(sampled)) * prob)
    else:
        states.append(board.copy())
        probs.append(1.0)

    # 2. 考场意外
    if step_count % 8 == 0 and step_count > 0:
        new_states, new_probs = [], []
        for base_state, base_prob in zip(states, probs):
            cand = [(r, c) for r in range(size) for c in range(size)
                    if 4 <= base_state[r, c] < 128]   # 只降级国三及以下
            if not cand:
                new_states.append(base_state)
                new_probs.append(base_prob)
            else:
                if len(cand) > ACCIDENT_SAMPLE_MAX:
                    cand = random.sample(cand, ACCIDENT_SAMPLE_MAX)
                for cell in cand:
                    nb = base_state.copy()
                    nb[cell] = max(2, nb[cell] // 2)
                    new_states.append(nb)
                    new_probs.append(base_prob / len(cand))
        states, probs = new_states, new_probs
    # 3. 金牌教练
    if step_count % 12 == 0 and step_count > 0:
        new_states, new_probs = [], []
        for base_state, base_prob in zip(states, probs):
            empty_coach = list(zip(*np.where(base_state == 0)))
            if not empty_coach:
                new_states.append(base_state)
                new_probs.append(base_prob)
            else:
                if len(empty_coach) > COACH_SAMPLE_MAX:
                    empty_coach = random.sample(empty_coach, COACH_SAMPLE_MAX)
                for cell in empty_coach:
                    for tile_val, prob in COACH_TILE_PROBS:
                        nb = base_state.copy()
                        nb[cell] = tile_val
                        new_states.append(nb)
                        new_probs.append(base_prob * (1.0 / len(empty_coach)) * prob)
        states, probs = new_states, new_probs

    expected = 0.0
    for state, prob in zip(states, probs):
        max_val = max_node(state, step_count, depth)
        expected += prob * max_val

    trans_table[key] = expected
    return expected


def max_node(board, step_count, depth):
    best = -float('inf')
    actions = list(range(4))
    action_scores = []
    for a in actions:
        nb, moved, ms = simulate_move(board, a)
        if moved:
            fast_val = evaluate(nb) + ms
            action_scores.append((a, nb, ms, fast_val))
    if not action_scores:
        return evaluate(board)

    action_scores.sort(key=lambda x: x[3], reverse=True)
    candidates = action_scores[:2]

    for a, nb, ms, _ in candidates:
        future = chance_node(nb, step_count+1, depth-1)
        value = ms + future
        if value > best:
            best = value
    return best


def get_best_action(game, depth=3):
    board = game.board.copy()
    step_count = game.step_count
    best_action = None
    best_score = -float('inf')

    action_candidates = []
    for a in range(4):
        nb, moved, ms = simulate_move(board, a)
        if moved:
            fast_val = evaluate(nb) + ms
            action_candidates.append((a, nb, ms, fast_val))
    if not action_candidates:
        return None

    action_candidates.sort(key=lambda x: x[3], reverse=True)
    candidates = action_candidates[:2]

    for a, nb, ms, _ in candidates:
        future = chance_node(nb, step_count+1, depth-1)
        value = ms + future
        if value > best_score:
            best_score = value
            best_action = a
    return best_action
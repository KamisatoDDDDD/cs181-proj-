import numpy as np

# ==================== 各版本权重 ====================
# 你可以直接修改这些数字来调参

WEIGHTS_V0 = {
    'empty': 1.0,   # 仅空格数，无需其他权重
}

WEIGHTS_V1 = {
    'empty': 35.0,
    'corner_max': 12.0,
}

WEIGHTS_V2 = {
    'empty': 35.0,
    'corner_max': 12.0,
    'smoothness': -1.5,
}

WEIGHTS_V3 = {
    'empty': 35.0,
    'monotonicity': -1.2,
    'smoothness': -1.5,
    'corner_max': 12.0,
}

WEIGHTS_V4 = {
    'empty': 35.0,
    'monotonicity': -1.2,
    'smoothness': -1.5,
    'corner_max': 12.0,
    'snake': 0.8,            # 蛇形权重
    'merge_potential': 2.5,  # 高层合并潜力
}

# 蛇形权重矩阵（5×5）
SNAKE_WEIGHTS = np.array([
    [16, 15, 14, 13, 12],
    [ 7,  8,  9, 10, 11],
    [ 6,  5,  4,  3,  2],
    [ 1,  2,  3,  4,  5],
    [ 0,  1,  2,  3,  4]
], dtype=np.float32)


def _monotonicity_score(arr):
    s = 0
    for i in range(len(arr)-1):
        if arr[i] >= arr[i+1]:
            s += 1
        else:
            s -= 1
    return s


# ==================== 五个评估函数 ====================

def evaluate_v0(board):
    """V0：仅空格数"""
    return WEIGHTS_V0['empty'] * np.sum(board == 0)


def evaluate_v1(board):
    """V1：空格数 + 角落最大值"""
    size = board.shape[0]
    empty = np.sum(board == 0)
    corners = [board[0, 0], board[0, size-1], board[size-1, 0], board[size-1, size-1]]
    max_corner = max(corners)
    max_val = np.max(board)
    if max_corner == max_val:
        corner_score = max_val
    else:
        corner_score = -max_val * 0.5
    return WEIGHTS_V1['empty'] * empty + WEIGHTS_V1['corner_max'] * corner_score


def evaluate_v2(board):
    """V2：空格数 + 角落最大值 + 平滑度"""
    size = board.shape[0]
    log_board = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    empty = np.sum(board == 0)
    corners = [board[0, 0], board[0, size-1], board[size-1, 0], board[size-1, size-1]]
    max_corner = max(corners)
    max_val = np.max(board)
    if max_corner == max_val:
        corner_score = max_val
    else:
        corner_score = -max_val * 0.5

    smooth = 0.0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] > 0 and board[r, c+1] > 0:
                smooth -= abs(log_board[r, c] - log_board[r, c+1])
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] > 0 and board[r+1, c] > 0:
                smooth -= abs(log_board[r, c] - log_board[r+1, c])

    return (WEIGHTS_V2['empty'] * empty +
            WEIGHTS_V2['corner_max'] * corner_score +
            WEIGHTS_V2['smoothness'] * smooth)


def evaluate_v3(board):
    """V3：空格数 + 角落最大值 + 平滑度 + 单调性（当前完整版）"""
    size = board.shape[0]
    log_board = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    empty = np.sum(board == 0)
    corners = [board[0, 0], board[0, size-1], board[size-1, 0], board[size-1, size-1]]
    max_corner = max(corners)
    max_val = np.max(board)
    if max_corner == max_val:
        corner_score = max_val
    else:
        corner_score = -max_val * 0.5

    smooth = 0.0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] > 0 and board[r, c+1] > 0:
                smooth -= abs(log_board[r, c] - log_board[r, c+1])
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] > 0 and board[r+1, c] > 0:
                smooth -= abs(log_board[r, c] - log_board[r+1, c])

    mono = 0.0
    for r in range(size):
        row = log_board[r]
        mono += max(_monotonicity_score(row), _monotonicity_score(row[::-1]))
    for c in range(size):
        col = log_board[:, c]
        mono += max(_monotonicity_score(col), _monotonicity_score(col[::-1]))

    return (WEIGHTS_V3['empty'] * empty +
            WEIGHTS_V3['corner_max'] * corner_score +
            WEIGHTS_V3['smoothness'] * smooth +
            WEIGHTS_V3['monotonicity'] * mono)


def evaluate_v4(board):
    """V4：蛇形至尊版 = V3 + 蛇形权重 + 高层合并潜力"""
    size = board.shape[0]
    log_board = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    empty = np.sum(board == 0)
    corners = [board[0, 0], board[0, size-1], board[size-1, 0], board[size-1, size-1]]
    max_corner = max(corners)
    max_val = np.max(board)
    if max_corner == max_val:
        corner_score = max_val
    else:
        corner_score = -max_val * 0.5

    smooth = 0.0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] > 0 and board[r, c+1] > 0:
                smooth -= abs(log_board[r, c] - log_board[r, c+1])
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] > 0 and board[r+1, c] > 0:
                smooth -= abs(log_board[r, c] - log_board[r+1, c])

    mono = 0.0
    for r in range(size):
        row = log_board[r]
        mono += max(_monotonicity_score(row), _monotonicity_score(row[::-1]))
    for c in range(size):
        col = log_board[:, c]
        mono += max(_monotonicity_score(col), _monotonicity_score(col[::-1]))

    snake = np.sum(log_board * SNAKE_WEIGHTS)

    merge_pot = 0.0
    for r in range(size):
        for c in range(size - 1):
            if board[r, c] == board[r, c+1] and board[r, c] >= 16:
                merge_pot += log_board[r, c]
    for c in range(size):
        for r in range(size - 1):
            if board[r, c] == board[r+1, c] and board[r, c] >= 16:
                merge_pot += log_board[r, c]

    return (WEIGHTS_V4['empty'] * empty +
            WEIGHTS_V4['corner_max'] * corner_score +
            WEIGHTS_V4['smoothness'] * smooth +
            WEIGHTS_V4['monotonicity'] * mono +
            WEIGHTS_V4['snake'] * snake +
            WEIGHTS_V4['merge_potential'] * merge_pot)
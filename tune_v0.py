import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V0, evaluate_v0

VERSION = "V0"
TUNE_DEPTH = 2               # 调参用浅层搜索，比最终实验的 depth=3 快 5~10 倍
NUM_TUNE_GAMES = 10
MAX_STEPS = 200              # 强制终止：避免低效策略无休止地苟活
LOG_FILE = f"results/tune_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

# 缩小搜索范围（粗搜），后续如需精调可以基于粗搜结果缩小范围
SEARCH_SPACE = {
    'empty': [20, 40, 60, 80, 100]    # 原本 10 个值减为 5 个，涵盖典型区间
}

def evaluate_params(empty_val):
    WEIGHTS_V0['empty'] = empty_val
    scores = []
    for seed in range(NUM_TUNE_GAMES):
        random.seed(seed)
        np.random.seed(seed)
        game = Game()
        game.reset()
        step = 0
        while step < MAX_STEPS:                    # 步数保护
            action = get_best_action(game, depth=TUNE_DEPTH, eval_func=evaluate_v0)
            if action is None:
                break
            _, _, done, _ = game.step(action)
            step += 1
            if done:
                break
        scores.append(game.score)
    return np.mean(scores)

def main():
    os.makedirs("results", exist_ok=True)
    best_score = -float('inf')
    best_weight = None
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['empty', 'avg_score'])
        for empty_val in SEARCH_SPACE['empty']:
            avg = evaluate_params(empty_val)
            writer.writerow([empty_val, f"{avg:.2f}"])
            print(f"empty={empty_val} => avg={avg:.2f}")
            if avg > best_score:
                best_score = avg
                best_weight = {'empty': empty_val}
    print(f"\n{VERSION} 最佳权重: {best_weight}, 平均得分: {best_score:.2f}")
    with open(BEST_FILE, 'w') as fp:
        json.dump(best_weight, fp, indent=2)

if __name__ == '__main__':
    main()
import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V1, evaluate_v1

VERSION = "V1"
TUNE_DEPTH = 2               # 调参用浅层搜索，比最终实验的 depth=3 快很多
NUM_TUNE_GAMES = 10
MAX_STEPS = 200              # 强制终止，避免低效策略无限拖延
LOG_FILE = f"results/tune_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

SEARCH_SPACE = {
    'empty': [20, 30, 40, 50],
    'corner_max': [5, 10, 15, 20]
}

def evaluate_params(empty_val, corner_val):
    WEIGHTS_V1['empty'] = empty_val
    WEIGHTS_V1['corner_max'] = corner_val
    scores = []
    for seed in range(NUM_TUNE_GAMES):
        random.seed(seed)
        np.random.seed(seed)
        game = Game()
        game.reset()
        step = 0
        while step < MAX_STEPS:
            action = get_best_action(game, depth=TUNE_DEPTH, eval_func=evaluate_v1)
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
    best_weights = None
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['empty', 'corner_max', 'avg_score'])
        for empty_val in SEARCH_SPACE['empty']:
            for corner_val in SEARCH_SPACE['corner_max']:
                avg = evaluate_params(empty_val, corner_val)
                writer.writerow([empty_val, corner_val, f"{avg:.2f}"])
                print(f"empty={empty_val}, corner_max={corner_val} => avg={avg:.2f}")
                if avg > best_score:
                    best_score = avg
                    best_weights = {'empty': empty_val, 'corner_max': corner_val}
    print(f"\n{VERSION} 最佳权重: {best_weights}, 平均得分: {best_score:.2f}")
    with open(BEST_FILE, 'w') as fp:
        json.dump(best_weights, fp, indent=2)

if __name__ == '__main__':
    main()
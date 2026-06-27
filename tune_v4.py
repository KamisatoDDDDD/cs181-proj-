import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V4, evaluate_v4

VERSION = "V4"
TUNE_DEPTH = 2               # 调参用浅层搜索，比最终实验的 depth=3 快很多
NUM_TUNE_GAMES = 10
MAX_STEPS = 200              # 强制终止，避免低效策略无限拖延
LOG_FILE = f"results/tune_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

SEARCH_SPACE = {
    'empty': [20, 35, 50],
    'monotonicity': [-0.8, -1.2],
    'smoothness': [-1.0, -1.5],
    'corner_max': [10, 15],
    'snake': [0.5, 1.0],
    'merge_potential': [1.5, 3.0]
}

def evaluate_params(empty_val, mono_val, smooth_val, corner_val, snake_val, merge_val):
    WEIGHTS_V4['empty'] = empty_val
    WEIGHTS_V4['monotonicity'] = mono_val
    WEIGHTS_V4['smoothness'] = smooth_val
    WEIGHTS_V4['corner_max'] = corner_val
    WEIGHTS_V4['snake'] = snake_val
    WEIGHTS_V4['merge_potential'] = merge_val
    scores = []
    for seed in range(NUM_TUNE_GAMES):
        random.seed(seed)
        np.random.seed(seed)
        game = Game()
        game.reset()
        step = 0
        while step < MAX_STEPS:
            action = get_best_action(game, depth=TUNE_DEPTH, eval_func=evaluate_v4)
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
        writer.writerow(['empty','monotonicity','smoothness','corner_max','snake','merge_potential','avg_score'])
        for empty_val in SEARCH_SPACE['empty']:
            for mono_val in SEARCH_SPACE['monotonicity']:
                for smooth_val in SEARCH_SPACE['smoothness']:
                    for corner_val in SEARCH_SPACE['corner_max']:
                        for snake_val in SEARCH_SPACE['snake']:
                            for merge_val in SEARCH_SPACE['merge_potential']:
                                avg = evaluate_params(empty_val, mono_val, smooth_val, corner_val, snake_val, merge_val)
                                writer.writerow([empty_val, mono_val, smooth_val, corner_val,
                                                 snake_val, merge_val, f"{avg:.2f}"])
                                print(f"empty={empty_val}, mono={mono_val}, smooth={smooth_val}, "
                                      f"corner={corner_val}, snake={snake_val}, merge={merge_val} => avg={avg:.2f}")
                                if avg > best_score:
                                    best_score = avg
                                    best_weights = {'empty': empty_val, 'monotonicity': mono_val,
                                                    'smoothness': smooth_val, 'corner_max': corner_val,
                                                    'snake': snake_val, 'merge_potential': merge_val}
    print(f"\n{VERSION} 最佳权重: {best_weights}, 平均得分: {best_score:.2f}")
    with open(BEST_FILE, 'w') as fp:
        json.dump(best_weights, fp, indent=2)

if __name__ == '__main__':
    main()
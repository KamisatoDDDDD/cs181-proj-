import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V0, evaluate_v0

VERSION = "V0"
DEPTH = 3
NUM_TUNE_GAMES = 10
LOG_FILE = f"results/tune_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

SEARCH_SPACE = {
    'empty': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

def evaluate_params(empty_val):
    WEIGHTS_V0['empty'] = empty_val
    scores = []
    for seed in range(NUM_TUNE_GAMES):
        random.seed(seed)
        np.random.seed(seed)
        game = Game()
        game.reset()
        while True:
            action = get_best_action(game, depth=DEPTH, eval_func=evaluate_v0)
            if action is None:
                break
            _, _, done, _ = game.step(action)
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
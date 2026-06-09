import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V3, evaluate_v3

VERSION = "V3"
DEPTH = 3
NUM_GAMES = 100
LOG_FILE = f"results/final_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

def load_best_params():
    with open(BEST_FILE, 'r') as f:
        return json.load(f)

def main():
    best = load_best_params()
    # V3 参数: empty, monotonicity, smoothness, corner_max
    WEIGHTS_V3['empty'] = best['empty']
    WEIGHTS_V3['monotonicity'] = best['monotonicity']
    WEIGHTS_V3['smoothness'] = best['smoothness']
    WEIGHTS_V3['corner_max'] = best['corner_max']
    print(f"{VERSION} 使用权重: {best}")

    os.makedirs("results", exist_ok=True)
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game_id', 'score', 'max_tile', 'steps', 'duration_sec'])
        for game_id in range(1, NUM_GAMES+1):
            random.seed(game_id)
            np.random.seed(game_id)
            game = Game()
            game.reset()
            start = time.time()
            while True:
                action = get_best_action(game, depth=DEPTH, eval_func=evaluate_v3)
                if action is None:
                    break
                _, _, done, _ = game.step(action)
                if done:
                    break
            duration = time.time() - start
            max_tile = game.board.max()
            print(f"{VERSION} 第{game_id}局 | 得分:{game.score} | "
                  f"最高等级:{max_tile} | 步数:{game.step_count} | 耗时:{duration:.1f}s")
            writer.writerow([game_id, game.score, max_tile, game.step_count, duration])

if __name__ == '__main__':
    main()
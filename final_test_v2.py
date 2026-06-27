import random, json, os, csv, time
import numpy as np
from environment import Game
from expectimax import get_best_action
from evals import WEIGHTS_V2, evaluate_v2

VERSION = "V2"
DEPTH = 2
NUM_GAMES = 50
LOG_FILE = f"results/final_{VERSION.lower()}.csv"
BEST_FILE = f"best_{VERSION.lower()}.json"

def load_best_params():
    with open(BEST_FILE, 'r') as f:
        return json.load(f)

def main():
    best = load_best_params()
    # V2 参数: empty, corner_max, smoothness
    WEIGHTS_V2['empty'] = best['empty']
    WEIGHTS_V2['corner_max'] = best['corner_max']
    WEIGHTS_V2['smoothness'] = best['smoothness']
    print(f"{VERSION} 使用权重: {best}")

    os.makedirs("results", exist_ok=True)
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game_id', 'score', 'max_tile', 'steps', 'duration_sec'])
        for game_id in range(12, NUM_GAMES+12):
            random.seed(game_id)
            np.random.seed(game_id)
            game = Game()
            game.reset()
            start = time.time()
            while True:
                action = get_best_action(game, depth=DEPTH, eval_func=evaluate_v2)
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
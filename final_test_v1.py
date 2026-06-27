#!/usr/bin/env python3
import random, json, os, csv, time, argparse
import numpy as np
from environment import Game
import expectimax
from evals import WEIGHTS_V4, evaluate_v4

VERSION = "V1"
DEPTH = 2
BEST_FILE = "best_v4.json"

# 粗糙采样配置
expectimax.SAMPLE_THRESHOLD = 3
expectimax.SAMPLE_SIZE = 3
expectimax.ACCIDENT_SAMPLE_MAX = 2
expectimax.COACH_SAMPLE_MAX = 2

def load_best_params():
    with open(BEST_FILE, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_start', type=int, required=True)
    parser.add_argument('--num_games', type=int, default=30)
    args = parser.parse_args()

    best = load_best_params()
    for k, v in best.items():
        WEIGHTS_V4[k] = v
    print(f"{VERSION} 使用权重 (来自 V4): {best}")

    os.makedirs("results", exist_ok=True)
    seed_end = args.seed_start + args.num_games - 1
    log_file = f"results/final_{VERSION.lower()}_{args.seed_start}_{seed_end}.csv"

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game_id', 'score', 'max_tile', 'steps', 'duration_sec'])
        for game_id in range(args.seed_start, args.seed_start + args.num_games):
            random.seed(game_id)
            np.random.seed(game_id)
            game = Game()
            game.reset()
            start = time.time()
            while True:
                action = expectimax.get_best_action(game, depth=DEPTH, eval_func=evaluate_v4)
                if action is None:
                    break
                _, _, done, _ = game.step(action)
                if done:
                    break
            duration = time.time() - start
            max_tile = game.board.max()
            print(f"[{VERSION}] 第{game_id}局 | 得分:{game.score} | "
                  f"最高等级:{max_tile} | 步数:{game.step_count} | 耗时:{duration:.1f}s")
            writer.writerow([game_id, game.score, max_tile, game.step_count, duration])
    print(f"[{VERSION}] 结果已保存至 {log_file}")

if __name__ == '__main__':
    main()
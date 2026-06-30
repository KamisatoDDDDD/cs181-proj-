import numpy as np
import random
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from environment import Game
from final_rl import FastRL
from expectimax import get_best_action


# =========================
# RL policy（安全版：不使用step rollback）
# =========================
def rl_policy(game, agent):

    best_action = None
    best_score = -1e9

    for a in range(4):

        nb, moved, reward = None, False, 0

        # 用simulate_move（必须依赖你已有函数）
        from expectimax import simulate_move
        nb, moved, reward = simulate_move(game.board, a)

        if not moved:
            continue

        value = agent.value(nb)
        score = reward + value

        if score > best_score:
            best_score = score
            best_action = a

    return best_action


# =========================
# Expectimax policy（V4 baseline）
# =========================
def ex_policy(game, depth):
    return get_best_action(game, depth=depth)


# =========================
# run single game（完全对齐seed）
# =========================
def run_game(seed, mode, agent, depth=2):

    random.seed(seed)
    np.random.seed(seed)

    game = Game()
    game.reset()

    start = time.time()

    while True:

        if mode == "rl":
            action = rl_policy(game, agent)
        else:
            action = ex_policy(game, depth)

        if action is None:
            break

        _, _, done, _ = game.step(action)

        if done:
            break

    return {
        "score": game.score,
        "steps": game.step_count,
        "time": time.time() - start,
        "max_tile": int(game.board.max())
    }

def plot_tile_pie(tile_bins, mode, num_games):

    labels = list(tile_bins.keys())
    sizes = list(tile_bins.values())

    # 颜色（固定，论文风格）
    colors = ['#8ecae6', '#219ebc', '#ffb703', '#fb8500', '#8338ec']

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        radius=1.0
    )

    # =========================
    # 关键：外部标注 + 引导线 + 不重叠
    # =========================

    for i, w in enumerate(wedges):

        ang = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))

        # 标签位置（外侧）
        label_x = 1.35 * x
        label_y = 1.35 * y

        # 画引导线
        ax.annotate(
            f"{labels[i]}\n{sizes[i]} games",
            xy=(x, y),
            xytext=(label_x, label_y),
            ha='center',
            va='center',
            arrowprops=dict(arrowstyle='-', color='black', lw=1),
            fontsize=11
        )

    # =========================
    # 图例（颜色解释）
    # =========================
    ax.legend(
        wedges,
        labels,
        title="Max Tile",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )

    ax.set_title(f"Max Tile Distribution ({mode}, {num_games} games)")

    plt.tight_layout()
    plt.show()


# =========================
# evaluation
# =========================
def evaluate(seed_start, num_games, mode, depth):

    if mode == "rl":
        agent = FastRL()

        # ⭐⭐⭐ 关键：加载训练模型
        agent.w = np.load("model.npy")
    else:
        agent = None

    scores, stepss, times, tiles = [], [], [], []
    records = []

    for i in range(num_games):

        seed = seed_start + i

        result = run_game(seed, mode, agent, depth)
        records.append({
            "game_id": i,
            "seed": seed,
            "score": result["score"],
            "max_tile": result["max_tile"],
            "time": result["time"]
        })

        scores.append(result["score"])
        stepss.append(result["steps"])
        times.append(result["time"])
        tiles.append(result["max_tile"])

        print(
            f"[SEED {seed}] "
            f"score={result['score']}, "
            f"steps={result['steps']}, "
            f"time={result['time']:.3f}s, "
            f"max={result['max_tile']}"
        )
    df = pd.DataFrame(records)

    file_name = f"eval_{mode}_games{num_games}.xlsx"
    df.to_excel(file_name, index=False)

    print(f"Excel saved: {file_name}")

    print("\n===== FINAL RESULT =====")
    print("mode:", mode)
    print("games:", num_games)
    print("avg score:", np.mean(scores))
    print("std score:", np.std(scores))
    print("avg steps:", np.mean(stepss))
    print("avg time:", np.mean(times))
    print("avg max tile:", np.mean(tiles))

    # ====== max tile统计 ======
    tile_bins = {
            "<=128": 0,
            "256": 0,
            "512": 0,
            "1024": 0,
            "2048+": 0
        }

    for t in tiles:
            if t <= 128:
                tile_bins["<=128"] += 1
            elif t == 256:
                tile_bins["256"] += 1
            elif t == 512:
                tile_bins["512"] += 1
            elif t == 1024:
                tile_bins["1024"] += 1
            else:
                tile_bins["2048+"] += 1
    plot_tile_pie(tile_bins, mode, num_games)

        # ====== 饼图 ======
def plot_tile_pie(tile_bins, mode, num_games):

    labels = list(tile_bins.keys())
    sizes = list(tile_bins.values())

    colors = ['#8ecae6', '#219ebc', '#ffb703', '#fb8500', '#8338ec']

    fig, ax = plt.subplots(figsize=(9, 9))

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        radius=1.0
    )

    # =========================
    # ⭐ 关键改动1：强制分布角度（避免挤）
    # =========================
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    for i, w in enumerate(wedges):

        # wedge中心角
        ang = (w.theta2 + w.theta1) / 2

        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))

        # =========================
        # ⭐ 关键改动2：外圈“固定分层”
        # =========================
        # 上下自动拉开距离（避免重叠）
        offset_y = (i - len(labels)/2) * 0.15

        label_x = 1.6 * np.sign(x)
        label_y = 1.3 * y + offset_y

        ha = 'left' if x > 0 else 'right'

        ax.annotate(
            f"{labels[i]}: {sizes[i]}",
            xy=(x, y),
            xytext=(label_x, label_y),
            ha=ha,
            va='center',
            fontsize=11,
            arrowprops=dict(
                arrowstyle='-',
                lw=1,
                color='black'
            )
        )

    # =========================
    # legend（不会重叠的颜色说明）
    # =========================
    ax.legend(
    wedges,
    labels,
    title="Max Tile",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=len(labels),   # ⭐ 横向排列关键
    frameon=False
)

    ax.set_title(f"Max Tile Distribution ({mode}, {num_games} games)")

    plt.tight_layout()
    plt.show()

# =========================
# CLI
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="rl", choices=["rl", "ex"])
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_games", type=int, default=30)
    parser.add_argument("--depth", type=int, default=2)

    args = parser.parse_args()

    evaluate(
        seed_start=args.seed_start,
        num_games=args.num_games,
        mode=args.mode,
        depth=args.depth
            )
    
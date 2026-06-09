#这个是测试V3的评估函数哦
from environment import Game
from expectimax import get_best_action
from evals import evaluate_v3   
import time

def render(board, score, step):
    names = {
        0: '  .', 2: '省四', 4: '省三', 8: '省二', 16: '省一',
        32: '省队', 64: '国三', 128: '国二', 256: '国一',
        512: '国集', 1024: '国家队', 2048: '无敌了'
    }
    print(f"\n步数:{step} 得分:{score}")
    for row in board:
        print(" ".join(f"{names[c]:>5}" for c in row))

game = Game()
state = game.reset()
render(state, game.score, game.step_count)

while True:
    start = time.time()
    action = get_best_action(game, depth=3, eval_func=evaluate_v3)  
    if action is None:
        print("无合法动作，游戏结束")
        break
    print(f"AI选择: {['上','下','左','右'][action]} (耗时{time.time()-start:.2f}s)")
    next_state, reward, done, info = game.step(action)
    render(next_state, game.score, game.step_count)
    if done:
        print("游戏结束！")
        break
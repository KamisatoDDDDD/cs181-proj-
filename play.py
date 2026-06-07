from environment import Game  # 假设你的环境代码保存在 environment.py

def render(board, score, step_count):
    """打印棋盘、得分和步数"""
    # 等级名称映射（可选，纯数字也行，这里做美化）
    names = {
        0: '  .',
        2: '省四', 4: '省三', 8: '省二', 16: '省一',
        32: '省队', 64: '国三', 128: '国二', 256: '国一',
        512: '国集', 1024: '国家队', 2048: '无敌了'
    }
    print(f"\n===== 步数: {step_count} | 得分: {score} =====")
    for row in board:
        row_str = " ".join(f"{names.get(cell, str(cell)):>5}" for cell in row)
        print(row_str)
    print()

def main():
    game = Game()
    state = game.reset()
    render(state, game.score, game.step_count)

    action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}  # 上/下/左/右
    print("操作：W(上) S(下) A(左) D(右)，Q退出")

    while True:
        cmd = input("你的移动: ").strip().lower()
        if cmd == 'q':
            break
        if cmd not in action_map:
            print("无效按键，请用 W/S/A/D")
            continue

        action = action_map[cmd]
        next_state, reward, done, info = game.step(action)

        if not info.get('valid', True):
            print(">> 无效移动！棋盘未变化")
        else:
            print(f">> 合并得分: {reward}")

        render(next_state, game.score, game.step_count)

        if done:
            print("游戏结束！")
            break

if __name__ == '__main__':
    main()
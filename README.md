===============================================================================
  Synthesized National Team - 2048 AI 项目 README
=================================================

【项目概述】
本项目在一个非平稳的 5x5 2048 变体环境（包含“考场意外”和“教练指导”事件）中，
对比了 Expectimax 搜索算法与轻量级 TD-Learning 强化学习算法的性能。

===============================================================================
 一、文件结构及功能说明
=======================

1. 环境与核心逻辑

---

environment.py
    游戏核心环境类（Game）。定义了5x5棋盘，包含正常的滑动、合并逻辑，
    以及"Exam Accident（考场意外）"和"Coach Guidance（教练指导）"两个周期性事件。

evals.py
    所有的启发式评估函数（V0 ~ V4）。
    包含 Basic, Mono, Full 三种启发式家族的权重定义及计算逻辑（如平滑度、单调性、
    Snake权重、合并潜力）。

expectimax.py
    Expectimax 搜索算法核心代码。
    包含 Max节点、Chance节点（含采样逻辑）、置换表（Transposition Table）以及
    外部调用的 get_best_action 接口。

2. 调参脚本（生成权重文件）

---

tune_v2.py
    用于对 V2 (Basic-F) 的权重进行网格搜索调参。
    输出：results/tune_v2.csv 和 best_v2.json。

tune_v3.py
    用于对 V3 (Mono-F) 的权重进行网格搜索调参。
    输出：results/tune_v3.csv 和 best_v3.json。

tune_v4.py
    用于对 V4 (Full-F) 的权重进行网格搜索调参。
    输出：results/tune_v4.csv 和 best_v4.json。

最佳权重文件（生成结果）：
best_v2.json, best_v3.json, best_v4.json
    存储了调参得到的最佳权重组合，供后续测试脚本读取。

3. 训练脚本

---

final_rl.py
    TD-Lin (RL) 基线模型的训练脚本。
    使用 3000 个 episode 训练线性价值网络。
    输出：model.npy （保存训练好的权重）。

4. 测试/评估脚本

---

final_test_v0.py
    测试 V0 (Mono-C) 代理。内部强制使用粗采样（C配置），加载 V3 的权重。

final_test_v1.py
    测试 V1 (Full-C) 代理。内部强制使用粗采样（C配置），加载 V4 的权重。

final_test_v2.py
    测试 V2 (Basic-F) 代理。使用默认细采样（F配置），加载 V2 的权重。

final_test_v3.py
    测试 V3 (Mono-F) 代理。使用默认细采样（F配置），加载 V3 的权重。

final_test_v4.py
    测试 V4 (Full-F) 代理。使用默认细采样（F配置），加载 V4 的权重。

final_rl_test_figure.py
    测试 TD-Lin (RL) 代理，并生成测试结果Excel文件以及最大方块饼图。

5. 游戏演示与结果绘图

---

play.py
    人机交互脚本。允许人类玩家通过 W/A/S/D 键手动玩该5x5环境。

analyze_report_plots_v2.py
    报告绘图脚本。读取 results 文件夹下的所有测试 CSV 文件，生成论文所需的箱线图、
    最大方块分布堆叠图、性能-代价权衡图等。
    输出：report_assets/ 文件夹下的图表文件。

6. 论文报告

---

ai_proj (1).pdf
    项目的最终实验报告（论文正文）。

===============================================================================
 二、完整实验流程（从调参到生成图表）
=====================================

【步骤 1】 启发式权重调参（Expectimax）
---------------------------------------

注：需要在当前目录下创建空的 results/ 文件夹，脚本会自动生成。

在终端依次运行以下命令：
  python tune_v2.py
  python tune_v3.py
  python tune_v4.py

这将开始对 V2/V3/V4 的权重进行网格搜索（depth=2, 10局, 200步截断）。
运行完毕后，当前目录下会生成 best_v2.json, best_v3.json, best_v4.json 文件。

【步骤 2】 训练 RL 智能体
-------------------------

运行以下命令：
  python final_rl.py

该脚本执行 3000 个 episode 的 TD 学习，训练完毕后会生成 model.npy 文件。

【步骤 3】 全量评估测试（生成 CSV 数据）
----------------------------------------

（注：为配合报告中的 180 局数据，建议使用 --seed_start 12 --num_games 180 参数）

3.1 评估 Expectimax 代理 (V0 ~ V4)
建议在终端依次执行以下 5 条指令（请确保当前目录下已有对应的 best_v*.json）：
  python final_test_v0.py --seed_start 12 --num_games 180
  python final_test_v1.py --seed_start 12 --num_games 180
  python final_test_v2.py --seed_start 12 --num_games 180
  python final_test_v3.py --seed_start 12 --num_games 180
  python final_test_v4.py --seed_start 12 --num_games 180

运行完毕后，results/ 目录下会生成 final_v0_12_191.csv 等 5 个文件。（实际我们三人分工，所以这些文件是截断的）

3.2 评估 TD-Lin 代理
运行以下命令：
  python final_rl_test_figure.py --mode rl --seed_start 12 --num_games 180

运行完毕后，results/ 目录下会生成 eval_rl_games180.xlsx 文件。

【步骤 4】 数据可视化与报告图表生成
-----------------------------------

确保 results/ 目录下已经包含了所有上述的 CSV 和 Excel 数据文件后，运行：

  python analyze_report_plots_v2.py --results-dir results --out-dir report_assets --seed-start 12 --num-games 180

脚本会自动将数据合并，并生成：

- report_assets/score_boxplot_log.png      （分数分布箱线图）
- report_assets/max_tile_distribution_percent.png （最大方块分布图）
- report_assets/score_runtime_tradeoff_step.png （性能-成本权衡图）
- report_assets/merged_results.csv         （所有结果的聚合数据表）

【可选步骤 5】 人机对战测试
---------------------------

如果你想手动体验游戏，可运行：
  python play.py
使用 W/A/S/D 控制上下左右，Q 退出。

===============================================================================
 三、注意事项
=============

1. 测试脚本中的 --seed_start 和 --num_games 参数建议保持一致，以确保对照实验的公平性。
2. 粗采样（V0, V1）和细采样（V2, V3, V4）的配置代码直接硬编码在各自的 final_test_*.py 文件中，位于文件头部的 expectimax 模块参数赋值区。
3. 若缺少依赖库，请运行 pip install numpy pandas matplotlib openpyxl。

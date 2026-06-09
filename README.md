一、文件结构

合成国家队/

│

├── environment.py              # 游戏核心：5×5 棋盘，移动合并，随机生成，扰动，游戏结束判定

├── expectimax.py               # Expectimax 搜索框架（支持 eval\_func 切换，含置换表）

├── evals.py                    # V0\~V4 评估函数及对应的权重字典（可在此调参）

│

├── play.py              # 手动玩（这是我测试游戏环境写的，不用管）

│

├── tune\_v0.py                  # V0 调参脚本（网格搜索最优 empty 权重）

├── tune\_v1.py                  # V1 调参脚本（网格搜索 empty + corner\_max）

├── tune\_v2.py                  # V2 调参脚本（网格搜索 empty + corner\_max + smoothness）

├── tune\_v3.py                  # V3 调参脚本（网格搜索四个参数）

├── tune\_v4.py                  # V4 调参脚本（网格搜索六个参数）

│

├── final\_test\_v0.py            # V0 最终实验：读取 best\_v0.json，运行 100 局（固定随机种子）

├── final\_test\_v1.py            # V1 最终实验：读取 best\_v1.json，运行 100 局

├── final\_test\_v2.py            # V2 最终实验：读取 best\_v2.json，运行 100 局

├── final\_test\_v3.py            # V3 最终实验：读取 best\_v3.json，运行 100 局

├── final\_test\_v4.py            # V4 最终实验：读取 best\_v4.json，运行 100 局

│

├── results/                    # 存放所有调参和最终实验的 CSV 数据（这个是跑完tune和final\_text们会得到的结果，现在还没有）

│   ├── tune\_v0.csv

│   ├── tune\_v1.csv

│   ├── tune\_v2.csv

│   ├── tune\_v3.csv

│   ├── tune\_v4.csv

│   ├── final\_v0.csv

│   ├── final\_v1.csv

│   ├── final\_v2.csv

│   ├── final\_v3.csv

│   └── final\_v4.csv

│

├── best\_v0.json                # V0 调参得到的最优权重（这些是跑完tune们会得到的结果，现在还没有）

├── best\_v1.json                # V1 最优权重

├── best\_v2.json                # V2 最优权重

├── best\_v3.json                # V3 最优权重

├── best\_v4.json                # V4 最优权重

│

├── test\_v0.py                  # (可选) V0 单局演示脚本，人工观察 AI 行为

├── test\_v1.py

├── test\_v2.py

├── test\_v3.py

├── test\_v4.py

│

└── README.md                   # 项目说明（可写实验流程、运行方式等）



二、项目流程

整个项目分为 准备阶段 → 调参阶段 → 最终实验阶段 → 分析与撰写阶段。

**1. 准备阶段（代码实现与验证）**

(1) 编写 environment.py（已完成）。

(2) 编写 evals.py，实现 V0～V4 五个评估函数及权重字典。

(3) 编写 expectimax.py，实现能接受评估函数参数的搜索框架。

(4) 编写 test\_v0.py \~ test\_v4.py，各自调用对应评估函数跑一局，观察 AI 行为是否正常、有无错误。



**2. 调参阶段（寻找每个版本的最佳权重）**

(1) 三人分工运行 tune\_v0.py \~ tune\_v4.py。

(2) 每个调参脚本内预设了参数搜索空间(到时候得改一下)，对每种组合测试 NUM\_TUNE\_GAMES 局（默认 10 局），记录平均得分。

(3) 每局开始时固定 random.seed(seed) 和 np.random.seed(seed)，确保同一种子在不同权重下使用的随机序列完全一致（公平比较）。

(4) 脚本自动选择平均得分最高的权重组合，保存为 best\_v\*.json。



**3. 最终实验阶段（100 局控制变量对比）**

(1) 将所有人的 best\_v\*.json 汇总到项目根目录。

(2) 三人分别运行 final\_test\_v0.py \~ final\_test\_v4.py，每个脚本读取对应的最佳权重，运行 100 局。

(3) 每局使用 game\_id 作为随机种子（1～100），确保所有版本的第 1 局、第 2 局……第 100 局都在完全相同的初始棋盘、新方块出现位置、扰动触发序列下进行。

(4) 记录每局的最终得分、最高合成等级（max\_tile）、步数、耗时，写入 results/final\_v\*.csv。



**4. 数据分析与报告撰写阶段**

(1) 汇总 results/final\_v0.csv \~ final\_v4.csv。

(2) 用 Python（pandas + matplotlib/seaborn）读取数据，绘制图表：

(3) 平均得分柱状图（含标准误或置信区间; 最高合成等级分布（堆积条形图或百分比热力图; 生存步数箱线图; 单步决策耗时对比。

(4) 撰写实验报告：解释每个评估函数的设计思路，展示消融实验（V0→V4 逐步添加特征）的得分提升，结合图表分析原因。

(5) 讨论局限性（如深度限制、采样近似、权重调优空间）和未来工作（可与 RL 对比、引入更复杂启发式等）。




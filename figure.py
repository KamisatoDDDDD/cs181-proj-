import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 若英文可注释此行
plt.rcParams['axes.unicode_minus'] = False

# 版本列表
versions = ['v0', 'v1', 'v2', 'v3', 'v4', 'qlinear']  # 根据你的实际文件名调整

# 读取所有 CSV 并合并
all_data = {}
for ver in versions:
    # 找到所有包含该版本名的 CSV 文件
    files = [f for f in os.listdir('results') if f.startswith(f'final_{ver}') and f.endswith('.csv')]
    if not files:
        continue
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join('results', f))
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    all_data[ver] = df_all

# 如果有些版本缺失，可以手动补充或跳过

# 合并所有版本到一个 DataFrame 并添加 Version 列
df_master = pd.concat(all_data, names=['Version']).reset_index(level=0).reset_index(drop=True)
df_master.to_csv('results/merged_results.csv', index=False)
print("合并完成，已保存 merged_results.csv")

# 计算各版本的平均指标
summary = df_master.groupby('Version').agg(
    avg_score=('score', 'mean'),
    std_score=('score', 'std'),
    avg_max_tile=('max_tile', 'mean'),
    avg_steps=('steps', 'mean'),
    avg_duration=('duration_sec', 'mean'),
    count=('score', 'count')
).reset_index()
print(summary)

# 绘制平均得分柱状图
plt.figure(figsize=(8,5))
sns.barplot(data=summary, x='Version', y='avg_score', palette='viridis')
plt.ylabel('Average Score')
plt.title('Average Final Score by Version')
plt.tight_layout()
plt.savefig('results/avg_score_bar.png', dpi=150)
plt.show()

# 绘制得分箱线图
plt.figure(figsize=(10,6))
sns.boxplot(data=df_master, x='Version', y='score', palette='Set2')
plt.ylabel('Final Score')
plt.title('Score Distribution by Version')
plt.tight_layout()
plt.savefig('results/score_boxplot.png', dpi=150)
plt.show()

# 绘制最高等级分布 (max_tile 频次)
# 可自定义等级区间
bins = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096]
labels = ['≤32','64','128','256','512','1024','2048','4096']
df_master['tile_cat'] = pd.cut(df_master['max_tile'], bins=bins, labels=labels, right=True)
tile_dist = df_master.groupby(['Version', 'tile_cat']).size().unstack(fill_value=0)
tile_dist_pct = tile_dist.div(tile_dist.sum(axis=1), axis=0) * 100
tile_dist_pct.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Accent')
plt.ylabel('Percentage (%)')
plt.title('Maximum Tile Distribution by Version')
plt.tight_layout()
plt.savefig('results/tile_distribution.png', dpi=150)
plt.show()
# Football Transfer Pricing Marketing Analytics

本项目分析核心问题：

> How do football clubs reveal market preferences through transfer spending?

这不是纯体育数据分析，而是以营销分析视角理解足球转会价格。Notebook 将球员视为差异化市场资产，把转会价格理解为 4P 中的 Price 信号，并使用 STP 框架解释俱乐部如何通过支出表达偏好。

## 分析主线

1. Youth / Growth Positioning：年龄、潜力、潜力差。
2. League / Brand Positioning：联赛、顶级联赛、俱乐部声望。
3. Entertainment / Attention Positioning：速度、盘带、射门、技术动作、国际声望。

## 主要文件

- `transfer_analysis.ipynb`：升级后的主 notebook。
- `transfer_analysis_executed.ipynb`：已执行并保留输出的 notebook。
- `transfer_analysis.py`：从 notebook 同步导出的脚本版本。
- `charts/`：notebook 执行后生成的演示图表。
- `football_marketing_presentation.html`：课程展示页面。
- `football_marketing_charts.json`：展示页面使用的图表配置数据。
- `分析结果汇总.md`、`图表营销洞察说明.md`、`营销课程HTML汇报文案.md`：分析结论与汇报材料。

原始 CSV 数据文件体积较大，其中部分文件超过 GitHub 单文件限制，因此不纳入 Git 版本管理。如需重新运行完整分析，请将 `male_players.csv`、`male_teams.csv` 和 `Transfer Market Data.csv` 放回本目录。

## 分析模块

- 数据理解：字段、缺失值、价格偏态、类别分布。
- 特征工程：年龄组、角色分组、顶级联赛、log price、potential gap、attention score、brand score。
- 高级可视化：增长定位图、联赛品牌阶梯、娱乐性定位图、定位矩阵。
- PCA：识别球员市场原型。
- t-SNE：展示球员身份如何形成市场邻域。
- 聚类分群：命名市场细分，如 Young Growth Assets、Elite Entertainers。
- 轻量 hedonic pricing model：解释 growth premium、brand premium、attention premium。
- 差异化竞争直觉：稀缺性溢价、联赛流动、年龄-角色溢价热力图。

## 运行方式

```bash
cd football_transfer_analysis
python -m jupyter notebook transfer_analysis.ipynb
```

如需重新执行并生成输出版本：

```bash
MPLCONFIGDIR=.matplotlib_cache python -m jupyter nbconvert \
  --to notebook --execute transfer_analysis.ipynb \
  --output transfer_analysis_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

## 依赖

见 `requirements.txt`。高级分析使用：

- pandas / numpy
- matplotlib / seaborn
- scikit-learn
- plotly
- jupyter

## 输出图表

图表会保存到 `charts/`，适合放入 10 分钟课堂展示或本科高阶分析 showcase。

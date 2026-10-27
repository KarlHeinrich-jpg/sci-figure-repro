# 各图设计要点与改造指引

本文件记录每张图的实现要点，方便把模板套用到自己的数据上。

---

## Fig. 1 机器学习流程图（`fig1_workflow.py`）

**结构**：外层细框 → 左栏「Data collection」+ 右栏「How machine learning works」→ 中间通栏标题带 → 底部 7 个算法缩略图。

**要点**

- 整页使用一个覆盖全图的坐标系（`xlim 0-100 / ylim 0-70`），所有图形按"版面百分比"摆放，改画布尺寸不会错位。
- 6 个流程圆用 `Circle` 统一半径，圆内图标直接在该坐标系里画（螺旋用同心 `Ellipse`，Y 形箭头用粗 `plot` + `Polygon` 箭头，SMOTE 用散点，SHAP 用彩色横条，混淆矩阵用 `Rectangle` 网格，模型选择用「示波器」轮廓）。
- 连线用 `link()` 封装：按左右方向自动吸附到圆的切线位置，避免水平线穿过圆形。
- 底部 7 个算法图各用一个 `inset_axes`（等价 `fig.add_axes`），局部坐标 `-3.4…3.4`，然后按需绘制网络/树/散点。

**改造**：换算法时只改底部 `panels` 段；换流程步骤时改 `pos` 字典与 `captions`。

---

## Fig. 2 雷达图 + 小提琴（`fig2_radar_violin.py`）

**要点**

- 雷达图用 `Polygon`（非 polar 轴）实现：五条辐条 + 10 层等距网格环 + 外框，便于精确控制每层环的形状与标签位置。
- 三重 `fill_betweenx` 小提琴：`common.violin()` 用 `gaussian_kde(bw_method=0.28)` 生成密度，虚线画出 Q1/中位数/Q3。
- 显著性括号 `common.bracket()`：横杠 + 两端下折，文字在上方居中；`p<0.001***` 作为 `ax.text` 放到坐标轴右上角。

**改造**：`attrs` 改维度名称，`vals` 改各组的 0–10 评分；小提琴把 `data` 换成自己的样本数组。

---

## Fig. 3 OPLS-DA + VIP（`fig3_oplsda_vip.py`）

**要点**

- 得分图：先算组内协方差主轴，再用 `Ellipse(width=2.9*s1, height=2.9*s2, angle=θ)` 画旋转置信椭圆，半透明填充 + 实心散点两层（散点 zorder 更高）。
- VIP 条形图：`barh` + `axvline(1.0, color='red')` 阈值线；化合物名逐一取自原文表 1 的 VIP 排序。

**改造**：`groups` 里替换每组中心/角度/长短轴；`comp`、`vip` 换成自己的变量与 VIP 值。

---

## Fig. 4 Lasso / 回归 / ROC / SHAP 八面板（`fig4_lasso_models.py`）

**要点**

- (a) 系数路径：按 `final + (v0-final)*sigmoid` 收缩，靠近 `xopt` 时归零或趋于稳定；未入选变量在各自 λ 处断崖归零。
- (b) 交叉验证曲线：`平均误差` 实线 + `lambda 误差` 点线，叠加散点示意各折结果。
- (c) 系数条形图：15 个特征横向 `barh`，正负分色同色（原文皆蓝）。
- (d)(h) 回归：灰色 calibration + 红色 prediction + 黑色拟合线；RMSE / R² 用 `ax.text` 放在左上角。
- (e) ROC：由目标 AUC 反推判别力 `mu`（正态假设），计算经验 ROC 后插值；11 条曲线颜色与图例逐条对应。
- (f)(g) SHAP：`beeswarm()` 手写蜂群——按 x 排序后按局部密度分层堆叠，颜色映射特征值；色条用 `ColorbarBase` + 上下端 "High/Low" 标注。

**改造**：把自己模型的 `coef_`、`roc_curve` 输出、`shap_values` 填进对应数据块即可（`shap` 包不是必需依赖）。

---

## Fig. 5 理化指标 + Mantel（`fig5_physchem_mantel.py`）

**要点**

- 7 个面板统一用 `violin()`，显著性标注分两行：上行 A–C，下行 A–B / B–C。
- Mantel 面板完全自绘：右上三角相关矩阵用边长随 |r| 变化的方块编码，左下用贝塞尔曲线连到「Score」节点；线型编码 p 值（实线 <0.05 / 虚线 ≥0.05），线宽编码 |r|，颜色编码正负相关。
- 图例四段（r.sign / p / r.abs / spearman's r）手工排版，色标用 `imshow` 渐变条模拟。

**改造**：`panels` 里改量纲、轴范围、均值；Mantel 面板改 `envs`、`corr`、`mantel` 三个数据结构。

---

## Fig. 6 LEfSe + 群落结构 + PCA（`fig6_lefse_pca.py`）

**要点**

- **进化分支图**自写递归布局：`cladogram()` 先深度遍历得到叶序，把叶均分到扇形角度区间，内部节点取子节点角度中点；每层半径固定并画 `Wedge` 色带，节点用白心圆，叶节点带罗马字标签（角度自动翻转保持可读）。
- 分类树与图例一一对应：`bact` / `fungi` 字典存层级、着色、子节点；`bact_leg` / `fungi_leg` 存图例编号与名称。
- 堆叠柱状图：21/17 个属，扇区颜色与图例颜色一致，柱间留白使分组（B-/A-/C-）清晰。
- PCA：三个样本簇 + 半透明椭圆，左上角标注 P 值与 R 值，右侧图例「Samples」。

**改造**：`cladogram` 可复用于任何 LEfSe 输出；把 `bact` 字典换成自己的 `lefse.res`（门纲目科属种 + LDA 分组）即可。

---

## 通用样式

`common.py` 里集中了：

- `rcParams`：Times New Roman、1.4 轴线宽、粗体轴标签；
- 三组配色：等级色（蓝/红/绿）、PCA 浅色、LEfSe 三类色；
- `violin()`、`bracket()`、`sig_ns_ns()`、`save()`。

想整体换风格（如中文期刊宋体、字号放大），只改这一个文件即可。

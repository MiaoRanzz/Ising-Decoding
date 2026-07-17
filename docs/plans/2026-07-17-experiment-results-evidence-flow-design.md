# 实验结果证据链与展现流程设计

日期：2026-07-17

## 1. 目标

重构 `lifelong_ood_model_training_technical_report.md` 的实验结果分析逻辑，使报告同时适合论文阅读和组会汇报。报告在引言和方法中将 SAGE-R9 架构与 sequential lifelong training 作为两项技术贡献介绍，在结果章节中则使用一条连续证据链验证完整方案 EvoSAGE-QEC。

中心问题为：

> EvoSAGE-QEC 是否同时具备目标任务解码能力、持续适应能力和未知复合噪声泛化能力？

当前结果不能提前声称架构与 sequential 存在严格协同效应。待 `Ising + sequential` 结果完成后，通过完整 `2x2` 对照估计架构主效应、训练范式主效应和交互项。

## 2. 备选结构与选择

评估过三种组织方式：

1. 主张、证据、边界：围绕完整方案逐步验证，主线最清晰。
2. 两项贡献并列：分别证明架构和训练范式，科学贡献清楚，但结果叙事容易割裂。
3. 基线逐级淘汰：适合工程选型，但弱化论文问题和机制归因。

最终采用融合方案：

- 引言、方法和实验设计使用“两项贡献并列”，明确架构贡献和训练范式贡献。
- 结果章节使用“主张、证据、边界”的单一证据链。
- 待 Ising-seq 完成后，在主证据之后加入 `2x2` 贡献归因，而不是重写整篇结果逻辑。

## 3. 六层门槛式证据链

后一层判断以前一层成立为前提。每层只回答一个问题。

### 3.1 基础解码能力

问题：SAGE-R9 是否值得作为完整系统的架构基础？

正文聚焦目标设置 `d=9、T0`，在相同数据、训练预算和推理协议下比较 Ising 与 SAGE-R9。主指标包括 LER、相对改善率、参数量、单轮 latency 和吞吐量。正文只用一句话说明跨距离边界，完整 `d=5/7/9` 结果放入附录，明确 `d=7` 收益延续、`d=5` 未复现优势，避免选择性披露。

### 3.2 持续演化能力

问题：sequential 训练能否连续吸收 T1-T4 的噪声知识，同时保持旧任务能力？

报告 forward adaptation、forgetting 和任务流平均 LER。no-EWC 是主要 sequential 方案。本层不展开 OOD，也不把 EWC 与主线并列；EWC 仅在消融层判断正则化是否带来额外价值。

### 3.3 完整方案的 OOD 主证据

问题：`SAGE-R9 + sequential no-EWC` 在固定复合 OOD 网格中是否整体优于合理基线？

只保留两项核心比较：

1. seq no-EWC 对比 SAGE-R9 T0 单环境训练；
2. seq no-EWC 对比 SAGE-R9 mixed-noise。

第一项判断 sequential 相对单环境训练的收益，第二项判断 sequential 是否超过反复访问全部已知噪声任务的强基线。

### 3.4 架构与训练范式贡献归因

问题：完整系统的收益来自架构、训练范式，还是二者交互？

使用以下 `2x2` 对照：

| 架构 | T0 单环境训练 | Sequential |
| --- | --- | --- |
| Ising | Ising-T0 | Ising-seq |
| SAGE-R9 | SAGE-T0 | SAGE-R9-seq |

报告四个直接效应：

- T0 下的架构效应：`SAGE-T0 - Ising-T0`
- sequential 下的架构效应：`SAGE-seq - Ising-seq`
- Ising 上的 sequential 效应：`Ising-seq - Ising-T0`
- SAGE-R9 上的 sequential 效应：`SAGE-seq - SAGE-T0`

交互项定义为：

```text
interaction =
(SAGE-seq - SAGE-T0)
- (Ising-seq - Ising-T0)
```

交互项为负且不确定性支持该方向时，才能声称 sequential 在 SAGE-R9 上带来更大的额外收益。四格未齐全前，该层显示“结果待完成”，不估计交互效应。

### 3.5 优势适用范围

问题：总体 OOD 优势在哪些条件下成立，在哪些条件下减弱？

分析顺序固定为：

1. distance：判断问题规模对收益的影响；
2. basis：判断 X/Z 非对称性；
3. multiplier：判断噪声强度趋势和饱和效应；
4. axis combination：判断优势与物理噪声类型的关系；
5. axis combination x multiplier：定位具体优势区和退化区。

这些分层只解释第三层的总体结论，不重复建立新的总体叙事。

### 3.6 反证、消融与工程代价

问题：方法何时失效，EWC 是否进一步有效，收益是否增加推理成本？

集中披露：

- seq 未胜出的配置和主要退化区域；
- `LER >= 0.49` 的饱和配置；
- seq + EWC 相对 seq no-EWC 的平均效果、配置稳定性和遗忘表现；
- 参数量、latency、吞吐量及相对 PyMatching 的 speedup。

Latency 只作为工程可用性证据，不作为 OOD 泛化证据。相同 SAGE-R9 架构的不同 checkpoint 若 latency 差异处于测量波动范围，只报告 sequential 未引入额外推理开销。

## 4. 每层的统一展现模板

每层严格采用：

> 问题 -> 公平对照 -> 效应量 -> 主表与主图 -> 一句话判断 -> 适用边界

图表数量受以下规则约束：

- 每层最多一张主表和一张主图；
- 后续分层不得重复第三层的总体均值；
- 正文不展示完整 297 配置明细；
- 相同数字只在首次支持主张的位置出现；
- 原始 LER 曲线、完整 paired 比较和 latency 明细进入附录。

建议主图如下：

1. 基础能力：`d=9` 的 LER-latency 二维图；
2. 持续演化：T0-T4 任务序列及 adaptation/forgetting 图；
3. OOD 主证据：相对 T0 和 mixed-noise 的 paired delta 森林图；
4. 贡献归因：四格模型效果和 interaction 图；
5. 适用范围：分层 delta 多面板图及 axis x multiplier 热图；
6. 边界与代价：未胜出区域、EWC 消融和效率汇总表。

## 5. 统计解释与结论判定

每项比较按三个层次解释。

### 5.1 总体效应

绝对效应定义为：

```text
delta = LER_candidate - LER_baseline
```

`delta < 0` 表示候选模型 LER 更低。相对改善率定义为：

```text
relative improvement =
(LER_baseline - LER_candidate) / LER_baseline * 100%
```

绝对差表示实际减少的逻辑错误率，相对改善率用于比较不同 LER 基线下的收益幅度。

### 5.2 跨配置一致性

同时报告：

- paired 95% CI：固定实验配置和样本协议下平均差异的采样精度；
- 胜出配置数：优势覆盖多少固定环境；
- 配置间 delta 标准差：收益对环境变化的敏感程度；
- 热图和最差分组：识别均值是否由少数配置驱动。

Pooled CI 很窄不代表任意未知噪声环境下都有效。

### 5.3 外推边界

结论只覆盖当前实验域：`d=5/7/9`、11 种训练轴组合、9 个固定倍率、X/Z basis 和既定噪声模拟器。若总体 delta 为负但某个分层为正，结论写为“总体有效但存在条件性退化”，不得写成全面优于。

每层结尾使用标准句式：

> 在【实验域】内，相对【基线】，模型在【主指标】上取得【效应量】；该优势在【主要分层】中保持，但在【边界条件】下未复现。

## 6. 数据流与组件边界

报告继续由聚合脚本生成，不手工维护数值。统一数据流为：

```text
推理 JSON
-> 配置级 paired 明细
-> 统一长表
-> 六层分析视图
-> 主表与图
-> Markdown 正文和附录
```

统一长表至少包含：

- architecture
- training_paradigm
- checkpoint
- distance
- basis
- axis_signature
- multiplier
- num_samples
- LER
- latency
- paired discordant counts
- delta
- confidence interval

六个分析视图只返回本层的表格数据、绘图数据和标准化结论字段。Markdown 层只负责章节排列和文本渲染，不重新计算统计量。

## 7. 缺失和异常处理

- Ising-seq 未完成时，第四层显示明确的待完成状态，不使用空值或零值替代。
- 只有四格模型具有相同配置、basis、样本数和 seed 时才计算 interaction。
- checkpoint、配置或结果缺失时，停止受影响的比较并列出缺失项。
- 配置数量或样本量不一致时禁止静默 pooled。
- 高 LER 饱和配置完整披露，但不用于放大模型优势。
- 重新聚合只能读取已有结果，不得启动推理或覆盖原始 JSON。

## 8. 验证与验收

### 8.1 统计测试

- 验证 delta、相对改善率和 interaction 公式；
- 验证 paired CI 使用同一批样本的 discordant counts；
- 验证 pooled 结果的配置集合和样本量一致；
- 验证饱和配置标记。

### 8.2 数据完整性测试

- 验证六层分析所需字段齐全；
- 验证每项公平对照的配置、basis、seed 和样本数一致；
- 验证 Ising-seq 缺失时不生成 interaction；
- 验证四格齐全后才替换待完成状态。

### 8.3 报告结构测试

- 验证六层章节顺序固定；
- 验证每层最多一张主表和一张主图；
- 验证 Markdown 图表引用存在；
- 验证正文不重复总体数值；
- 验证完整配置和 paired 明细保留在附录；
- 验证 latency 未被表述为泛化证据。

## 9. 成功标准

重构后的结果章节应使读者能够按顺序回答：

1. SAGE-R9 是否具备目标任务上的基础效果和效率？
2. sequential 是否实现了持续适应并控制遗忘？
3. 完整方案是否在固定复合 OOD 网格中超过 T0 和 mixed-noise？
4. 架构与训练范式分别贡献多少，是否存在交互？
5. 优势在哪些 distance、basis、倍率和噪声轴下成立？
6. EWC、饱和区和推理代价揭示了哪些方法边界？

任一结论均能追溯到唯一主表或主图，并明确区分总体效应、跨配置一致性和外推边界。

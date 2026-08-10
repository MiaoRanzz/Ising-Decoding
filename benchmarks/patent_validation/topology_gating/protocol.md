# 拓扑残差效用门控 v2 验证协议

## 1. 验证对象和证据边界

验证对象为：

`patent/一种基于拓扑残差效用门控的量子纠错神经预解码方法-交底书v2.md`

v2 的核心区别不是“对连通候选整簇接纳”，而是把连通簇作为联合决策域，
在域内生成合法组合，逐组合计算真实 GF(2) 残余和逻辑帧，再按状态相关效用
选择至多一个组合，并在接纳后重新评价后续候选域。

验证分为三层，结论不得跨层外推：

1. **内核正确性**：受控矩阵用例验证成簇、组合、冲突、GF(2) 试算、逻辑风险
   和状态重算。
2. **Phase 1／DEM 动作代理**：在表面码 DEM 采样上比较逐位置阈值、旧式整簇
   接纳和 v2 组合门控。该层验证核心门控机制，但 DEM 机制列不等同于真实四通道
   空间动作。
3. **Phase 2／真实四通道**：接入 Ising 模型的
   `data_z/data_x/measurement_x/measurement_z` 输出及其扩展 `H/L` 映射。
   只有完成该层，才能声称验证了 v2 的四类动作优选实施例。

`smoke` 只检查代码、数据、训练、门控、解码和输出链路，任何 smoke 数值均不得
用于专利效果结论。

## 2. 预注册假设

### H1：核心安全性

相对于相同模型、相同候选阈值下的逐位置提交，v2 组合门控的测试集逻辑错误率
差值，其成对 95% 置信区间上界不超过绝对非劣界 `0.002`。

### H2：后端工作量

满足 H1 后，v2 至少在一个预注册工作量指标上优于逐位置提交：残余探测器密度、
连通分量平方和或全局解码时延。端到端时延单独报告，不能用较低的全局解码时间
掩盖门控开销。

### H3：v2 相对旧式整簇决策

相对于 `whole_cluster_v1`，`combination_v2` 应在 LER 非劣的前提下，通过选择簇内
子集避免至少一类整簇提交造成的逻辑风险或残余恶化。受控单元测试必须证明该能力；
正式统计实验再判断其出现频率和净效果。

## 3. 数据分割与调参纪律

- train、validation、test 使用互不重叠的随机流。
- 候选阈值、效用权重和搜索预算只允许根据 validation 选择。
- test 只运行冻结后的参数；不得根据 test 结果回调阈值。
- 每个结果必须记录配置 SHA-256、Git commit、Git dirty 状态、Python/Torch 版本、
  设备、seed、shots 和证据范围。
- Phase 1 正式结果至少使用 3 个随机种子；Phase 2 至少覆盖两个码距、X/Z 两个
  逻辑基、登记噪声和漂移噪声。当前单 seed 配置只用于建立首轮流水线。

## 4. 对照组

| 名称 | 含义 |
|---|---|
| `raw_pymatching` | 不使用神经预解码的后端基准 |
| `bce_pointwise` | 在相同阈值和 Top-K 候选预算内直接提交全部候选 |
| `bce_whole_cluster_v1` | 候选连通分量作为整体接纳或拒绝的历史实现 |
| `bce_combination_v2` | 普通 BCE 模型加 v2 合法组合门控 |
| `topology_combination_v2` | 拓扑联合损失模型加相同 v2 门控 |

所有门控方法必须使用相同样本、同一模型概率和同一后端解码器进行成对比较。

## 5. 指标与判定顺序

1. 首先检查实际错误数、shots 和 LER 成对区间；错误事件不足时标记证据不足。
2. 只有满足 H1 的方法才比较残余密度、连通分量平方和和 PyMatching 时延。
3. 单独报告候选数、每 shot 评价组合数、精确搜索比例、接纳率和门控耗时。
4. 若使用预算束搜索，必须和小簇精确枚举结果做一致性审计；不得把近似搜索结果
   标记为全局最优组合。
5. 正式正向结论要求 H1 与 H2 同时成立；H3 和四通道结果作为 v2 特异性证据。

## 6. 冻结入口

配置：`conf/experiments/patent/topology_gating_v2.yaml`

CPU/GPU smoke：

```bash
PYTHONPATH=code conda run -n ising-decoding \
  /root/miniconda3/envs/ising-decoding/bin/python \
  -m benchmarks.patent_validation.topology_gating_v2 --mode smoke
```

Phase 1 单 seed pilot（用于冻结正式设计，不作为正式效果证据）：

```bash
PYTHONPATH=code conda run -n ising-decoding \
  /root/miniconda3/envs/ising-decoding/bin/python \
  -m benchmarks.patent_validation.topology_gating_v2 --mode pilot
```

Phase 1 单 seed 大样本运行（仍需与其他 seed 汇总后才可形成正式证据）：

```bash
PYTHONPATH=code conda run -n ising-decoding \
  /root/miniconda3/envs/ising-decoding/bin/python \
  -m benchmarks.patent_validation.topology_gating_v2 --mode validation --resume
```

输出位于 `outputs/patent_validation/topology_gating_v2/{smoke,pilot,validation}/`，不提交 Git。
正式归档需对至少 3 个独立 seed 重复运行、汇总统计并冻结结果目录校验和；当前入口
只执行配置中的一个 seed。

# QEWC V4 跨架构连续学习与泛化实验设计

> 2026-09-17 用户修订（覆盖下文分别选参规则）：停止 Ising 独立参数搜索，
> 两张 GPU 并行运行不同 HTNet 候选，保留当前正在运行的 HTNet lambda=0.1。
> 两架构统一使用 HTNet 选定的 lambda、Fisher 样本数、Replay 比例；gamma 保持 1。
> Fisher 张量仍由各自模型计算，Replay 容量仍匹配各自 EWC 状态字节数，不能跨架构复制。
> 已完成的 Ising 搜索结果仅作历史记录，不参与参数选择。Ising 正式 EWC/Replay
> 使用移植参数重新训练；这检验参数迁移，不再声称两架构分别达到最佳调参结果。

> 2026-09-11 用户修订：后续统一 B=30，覆盖预算 pilot 自动选择的 B=5。
> 详见 `budget_decision.json`。保留历史 epoch 和学习率调度，独立训练 30 轮，
> 不截取 E50 权重。每任务 15,360 次更新、62,914,560 实际样本。
> 补齐 HTNet 四个 E30 Oracle；两架构校准均使用 B=30。
> 不声称严格收敛；完整校准和正式矩阵预计超过原两周周期。
> 按用户要求滚动清理过程权重，保留最近两组完整断点及各阶段终点。

## 1. 研究问题

本实验检验：从已经完成 T0 训练的模型开始，在相同任务和训练预算下，Online EWC 是否比 SEQ 和 Replay 遗忘更少，并且在合成 OOD 与 Willow 真实硬件数据上都具有更好的泛化性能。

实验使用两种模型结构：

- HTNet A101，650,332 个参数；
- Ising-Fast，912,772 个参数。

同一结构内的 EWC、SEQ、Replay 对比用于判断 EWC 方法本身的作用。HTNet 与 Ising-Fast 的对比用于检查结论能否跨结构复现，但跨结构差异不能单独归因于 EWC。

本实验只有一个 T0 训练结果和一个 lifelong 训练重复，因此不声称结果跨随机初始化稳定。

## 2. 固定起点

主实验使用两个固定的 epoch-100 endpoint：

| 结构 | checkpoint | T0 seed |
|---|---|---:|
| HTNet A101 | `outputs/htnet_ablation_t0_a101_s4m_e100_seed12345_a100_skipup_v1/models/HTnet.0.100.pt` | 12345 |
| Ising-Fast | `outputs/ising_fast_qadapt_t0_s4m_e100_seed12345_a100_skipup_v1/models/PreDecoderModelMemory_v1.0.100.pt` | 12345 |

验证集最优的 HTNet epoch 97 和 Ising-Fast epoch 87 仅用于补充敏感性分析，不作为主起点。

两个历史 T0 模型都实际处理约 209,715,200 个样本。HTNet 完成 51,200 次优化器更新；Ising-Fast 因第一轮历史 batch schedule 完成 52,736 次更新。因此跨结构 T0 起点是样本量匹配，不是严格的优化器更新数匹配。T0 之后的全部训练统一使用固定 batch。

## 3. V4 中的 epoch 与预算

为了与历史 A101 实际执行结果保持一致，V4 使用“历史 A101 实际 epoch”：

- 配置请求 4,194,304 samples/epoch；
- 固定 micro-batch 为 2,048；
- gradient accumulation 为 2；
- 训练循环实际生成 2,097,152 samples/epoch；
- 每 epoch 完成 512 次 optimizer update；
- effective batch 为 4,096。

所有运行清单必须同时记录请求样本数、实际生成样本数、optimizer updates、GPU-hours 和峰值显存，不能只记录名义 epoch。

## 4. Stress 候选任务重新筛选

### 4.1 候选

筛选只使用 HTNet A101 epoch-100 endpoint。分别对 measurement、CNOT、idle、Z-bias 扫描 `1.5x、2x、3x、4x`，共 16 个候选。Ising-Fast 不参与正式任务选择。

### 4.2 候选测量

候选评测固定为 `d=9`、`rounds=9`、X/Z 两种基，每种基 32,768 shots。为隔离 epoch-97 到 epoch-100 的 checkpoint 变化，复用原 Stress 筛选随机流。

每个候选测量：

- HTNet epoch-100 的 Stage-0 LER；
- BCE/NLL 与 syndrome density；
- 1,024 个固定训练样本上的平均参数梯度；
- `head_activation` 的平均通道表示；
- 所有候选两两之间的梯度余弦和表示距离。

测量梯度时不更新模型参数。

### 4.3 倍率和顺序选择

从每类噪声选择一个候选，枚举全部 `4^4=256` 个组合：

1. 最小化四任务 Stage-0 LER 的极差；
2. 在最小极差之上允许 `0.005` 的绝对 LER 余量；
3. 合格组合中最小化六对任务的平均梯度余弦；
4. 再最大化六对任务的平均表示距离；
5. 最后按固定任务名顺序打破平局。

顺序首先选择与其他三个任务平均梯度余弦最低的任务，之后逐步选择与累计历史梯度余弦最低的任务。反向顺序是该顺序的完全反转。

选择完成后冻结倍率、两个顺序、配置、seed 和文件哈希。不得根据 EWC、OOD 或 Willow 结果修改。

### 4.4 Ising-Fast 诊断

Ising-Fast epoch-100 只评测 HTNet 已选出的四个任务，记录 LER、梯度余弦、表示距离和累计顺序冲突。诊断结果不改变任务或顺序。

## 5. 仅用 HTNet 选择每任务预算 B

HTNet 从 epoch-100 T0 endpoint 出发，对四个正式任务分别独立测试 `1、5、20、50` 个历史 A101 实际 epoch。如果 50 epoch 后仍未进入平台期，则扩展到 100。

不同预算必须分别从同一 T0 endpoint 开始，以免不同总长度的学习率调度互相混淆。

对每个任务选择最低充分预算：

- 增加到下一档后，固定验证流上的 LER 改善小于 0.1 个百分点；
- BCE 改善小于 1%；
- 训练和验证损失没有明显持续下降；
- 训练过程稳定。

最终 `B=max(B1,B2,B3,B4)`。HTNet 和 Ising-Fast 的所有正式方法均使用这个相同的 B。Ising-Fast 不参与 B 的选择。

HTNet 在 B 处的四个单任务模型作为 HTNet Oracle。Ising-Fast 只在固定 B 下训练四个单任务 Oracle，用于 Acquisition Gap 和预算充分性诊断；即使 Ising-Fast 更慢，也不提高正式预算。

## 6. EWC 参数重新选择

原 V3 的 `lambda=43.87539484535376` 被明确排除，不作为候选或默认值。

固定 Online EWC、per-sample BCE gradient 平方 Fisher、与训练一致的 positive weight、隔离的 run Fisher 状态和 `gamma=1`。gamma 衰减不属于核心矩阵。

Fisher 样本候选为 `4,096、16,384、65,536`。以 65,536 为参考，选择满足 Fisher 余弦不低于 0.99 且总量比例误差不超过 5% 的最小样本数。

HTNet 与 Ising-Fast 分别搜索 lambda，但使用相同规则：

- 粗搜索：`0.1、1、10、100、1000`；
- 若最优值在边界，向外扩展一个数量级；
- 在最优值与相邻值间加入几何中点进行细化。

参数选择只使用 T0-T4 ID 验证数据。候选先满足当前任务 LER 相对 SEQ 恶化不超过 5%、最终平均 LER 相对 SEQ 恶化不超过 2% 且训练稳定，再依次最小化 AF、使 BWT 更接近 0、最小化 FALER。OOD 和 Willow 不参与参数选择。

2026-09-21 用户在查看粗筛校准结果后授权将最终平均 LER 筛选门槛由 1% 调整为 2%。这是校准后的规则修订，不属于预先固定的原始规则；正式测试成功判据中的 1% 暂保持不变，不能将通过校准门槛等同于正式实验成功。

## 7. Replay 参数选择

两个结构分别比较 `10%、25%、50%` Replay 比例。Replay 样本包含在固定 batch 内，不额外增加 optimizer update。

每个结构的 Replay buffer 序列化字节数与该结构 EWC 的额外状态大小匹配。参数只根据 ID 验证集的 AF、FALER 和 GPU-hours 选择，不使用 OOD 或 Willow。

## 8. 正式训练矩阵

固定 T0 seed 为 `12345`，lifelong train seed 为 `20260907`。同一结构内的方法共享当前任务数据流。

每个结构在 HTNet 选出的主顺序上运行：

| 方法 | T0 后训练 | 预算 |
|---|---|---:|
| T0-only | 四个阶段继续训练 T0 | `4B` |
| SEQ | 顺序训练四个 Stress 任务 | `4B` |
| EWC | 相同顺序，使用新选参数 | `4B`，Fisher 另计 |
| Replay | 相同顺序，使用新选比例 | `4B` |
| Mix-total | T0-T4 等概率联合训练 | `4B` |
| Mix-exposure | T0-T4 等概率联合训练 | `5B` |

两个结构共 12 个主 run。每个结构另外运行 Reverse-SEQ 和 Reverse-EWC，共 4 个反向 run。正式训练总计 16 个 run。

T0-only 分成四个 B-epoch 阶段，每阶段重置 optimizer 和 scheduler，使阶段结构与 SEQ 一致。Mix-total 与主方法总训练预算相同，每个任务平均获得约 `0.8B`；Mix-exposure 每个任务平均获得约 `B`，但总预算多 25%。Mix 提前看到未来任务，只是联合训练参考线，不称为严格 lifelong 方法。

## 9. ID checkpoint 与指标

SEQ、EWC、Replay 固定保存：`C0=T0 epoch-100`，以及完成四个任务后的 `C1-C4`。每个 checkpoint 在 T0-T4 上评测，得到完整 `5x5` LER 矩阵。每个任务固定使用 X/Z 各 65,536 shots，所有方法使用相同配对测试流。

主指标：

- AF：历史任务平均遗忘，低好；
- BWT：后续任务对旧任务的影响，高好；
- AIP：整个训练过程平均成绩，高好；
- FALER：最终五任务平均 LER，低好；
- AG：持续学习是否阻碍新任务适应，接近 0 好。

主结果使用固定阶段 endpoint，不使用 OOD 或 Willow 选择 checkpoint。Mix 不报告严格 AF、BWT 或 AG。

## 10. OOD 与 Willow 共主要终点

两个评测同等重要，不合成一个总分，并从正式训练前冻结，不参与倍率、B、EWC、Replay 或 checkpoint 选择。

合成 OOD 复用现有 110 个配置、`d=7/9`、X/Z 每种基 262,144 shots 和相同配对流。报告宏平均、噪声类别、强度、码距、最差类别和方法配对差值。

Willow 复用现有 `d=5/7`、`rounds=10`、10 个案例、约 500,000 shots，全部零微调。保留 raw PyMatching，并分别比较同结构 EWC/SEQ/Replay/Mix、跨结构系统结果和端到端效果。

主顺序的全部最终模型及两个结构的 Reverse-SEQ/Reverse-EWC 都参加 OOD 与 Willow 评测。

## 11. 成功判据

在 HTNet 内，EWC lifelong 成功要求 AF 低于 SEQ、BWT 更接近 0、FALER 相对恶化不超过 1%、最后学习任务相对恶化不超过 5%，且 AG 不显示明显适应失败。Ising-Fast 同方向支持跨结构复现。

某个结构上的“EWC 总体泛化更好”要求：

- OOD 宏平均 LER 优于同结构 SEQ；
- Willow 宏平均 LER 优于同结构 SEQ；
- 两者相对 LER 改善均至少 1%；
- 配对 95% 区间均排除 0。

若只有一个外部数据集通过，只报告数据集特定改善。EWC 对 Replay 是次要比较；Mix 是联合训练参考；超过 raw PyMatching 是部署价值标准，不是 EWC lifelong 有效的必要条件。

单训练重复只能支持“固定训练重复下的一致证据”，不能声称跨训练 seed 稳定或普遍证明。

## 12. 资源记录与执行门禁

所有 run 记录实际样本、updates、GPU-hours、峰值显存、Fisher 时间、Replay 字节、模型参数、神经推理延迟、残差 PyMatching 延迟和端到端延迟。

执行顺序：

1. 冻结本设计和 manifest；
2. 增加并测试跨结构配置、launcher、审计与聚合入口；
3. HTNet epoch-100 重新筛选 Stress；
4. Ising-Fast 冲突诊断；
5. HTNet 预算 pilot 并冻结 B；
6. Ising-Fast Oracle；
7. 两结构 EWC 与 Replay 校准；
8. 16 个正式 run；
9. ID、OOD、Willow 配对评测；
10. 生成使用直白中文解释名词、设计、结果和限制的最终报告。

只有配置校验、dry-run 和短 smoke test 通过后，才允许启动长时间 GPU 训练。

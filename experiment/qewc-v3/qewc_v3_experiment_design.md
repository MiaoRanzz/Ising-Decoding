# QEWC v3 Lifelong Training 实验设计

## 1. 实验目标

本实验基于 A101-HTNet，在 T0–T4 噪声任务上比较四种训练范式：

1. **SEQ**：普通顺序微调；
2. **Online EWC**：参数重要性正则，无历史数据回放；
3. **Replay**：有限历史样本回放；
4. **Mix-noise**：可访问全部任务数据的联合训练上界。

实验需要回答：

1. 现实 T0–T4 噪声变化是否会导致灾难性遗忘；
2. 高冲突但物理合理的任务序列是否会导致遗忘；
3. EWC 能否减少遗忘；
4. EWC 是否因约束过强而无法学习新任务；
5. 相同额外存储下，EWC 与 Replay 谁更有效；
6. EWC 与 Mix 联合训练上界之间还有多大差距；
7. “没有遗忘”究竟来自知识保持，还是任务间正向迁移。

本轮固定一个训练 seed，定位为机制验证，不声称跨 seed 统计显著。

## 2. 共同 Stage-0

统一使用 A101-T0 epoch 97 权重：

```text
outputs/htnet_ablation_t0_a101_s4m_e100_seed12345_a100_skipup_v1/models/best_model/HTnet.0.97.pt
```

对应配置：

```text
conf/examples/qadapt/htnet_ablation/config_htnet_ablation_a101.yaml
```

固定 A101 架构：

```yaml
mixer: true
axis_channel_gate: false
input_skip: true
```

固定任务几何：

- distance：d=9；
- rounds：r=9；
- basis：X/Z both；
- observable/code rotation：沿用 A101-T0 配置；
- 网络输入和输出语义保持不变；
- T0–T4 只改变噪声模型。

所有方法必须：

- 仅加载 epoch 97 模型权重；
- 重置 optimizer、scheduler、scaler 和 epoch counter；
- 不继承原 T0 训练状态；
- 使用相同 continual training seed；
- 使用相同任务数据随机流；
- 使用独立输出目录和 Fisher 目录。

T0 作为已经学会的 Stage-0。SEQ、EWC、Replay 从 T1 开始，不重复训练 T0。

## 3. 任务设计

### 3.1 Realistic track

固定使用当前硬件含义明确的任务：

| 任务 | 噪声变化 |
|---|---|
| T0 | 国产硬件参数映射基础 25 参数噪声 |
| T1 | 仅 measurement 提升至 1.5× |
| T2 | 仅 CNOT 分量提升至 1.5× |
| T3 | 仅 idle-CNOT、idle-SPAM 提升至 1.5× |
| T4 | 指定 X/Z-bias 分量提升至 1.5× |

顺序固定为：

$$
T1\rightarrow T2\rightarrow T3\rightarrow T4.
$$

该轨道不要求出现遗忘。若 SEQ 不遗忘，应报告任务间干扰较弱或存在正向迁移，不能将其解释为 EWC 成功。

### 3.2 Stress track

分别对 measurement、CNOT、idle 和 Z-bias 扫描：

$$
1.5\times,\quad 2\times,\quad 3\times,\quad 4\times.
$$

对候选任务测量：

- syndrome density；
- A101-T0 LER；
- BCE/NLL；
- gradient cosine；
- representation distance。

正式任务选择原则：

1. 各任务单任务难度尽量接近；
2. 避免一个任务远难于其他任务；
3. 噪声机制存在明确差异；
4. 梯度冲突明显；
5. 不修改标签；
6. 不引入非物理噪声；
7. 不根据最终 EWC 结果反向修改任务。

生成两个顺序：

- **高冲突顺序**：逐步选择与累计历史梯度最冲突的下一任务；
- **反向顺序**：将高冲突顺序完全反转。

任务定义和顺序必须在正式训练前冻结。

## 4. 训练范式

### 4.1 SEQ

- 顺序训练 T1–T4；
- 不使用历史数据；
- 不使用 EWC；
- 不计算或加载 Fisher；
- 每个任务使用相同固定更新步数。

必须建立干净的 SEQ 入口，不能用 `EWC_LAMBDA=0` 模拟。

### 4.2 Online EWC

损失定义为：

$$
L=L_{\mathrm{data}}+
\frac{\lambda}{2}\sum_iF_i^*(\theta_i-\theta_i^*)^2.
$$

要求：

- 无 replay；
- 使用 online Fisher 累积；
- 主实验固定 $\gamma=1$；
- Fisher loss 与训练 loss 使用相同 `positive_weight`；
- 每阶段结束后估计 Fisher；
- 每个 run 隔离 Fisher；
- 禁止加载未来任务或其他实验的 Fisher；
- 记录每层 Fisher、EWC penalty 和参数漂移。

### 4.3 Replay

主实验只做 `SEQ+Replay`，本轮不做 `EWC+Replay`。

要求：

- Stage-0 建立固定 T0 buffer；
- 每完成一个任务，将固定样本加入 buffer；
- buffer 在历史任务间均衡分配；
- 不允许训练时无限重新生成旧任务；
- 总 batch size 和 optimizer steps 与 SEQ/EWC 相同；
- replay 样本替换部分当前任务样本，不额外扩大计算量。

Replay 比例在 pilot 中比较：

$$
10\%,\quad25\%,\quad50\%.
$$

正式训练前锁定，默认候选为 25%。

### 4.4 Mix-noise

Mix-noise 作为 joint-training oracle，不称为 lifelong 方法。

主实验使用 `Mix-total`：

- T0–T4 等权采样；
- 每个 batch 只来自一个任务；
- 总 optimizer steps 等于 SEQ/EWC 的 T1–T4 总步数；
- validation 分别计算五个任务的指标；
- checkpoint 选择参考五任务宏平均指标。

`Mix-exposure` 作为 stretch goal：

- 每个任务获得与 SEQ 单阶段相同的曝光量；
- 总计算量约为 SEQ 的 1.25 倍。

## 5. EWC 与 Replay 的公平性

A101 有 650,332 个参数。Online EWC 额外保存参考参数和 Fisher，按 FP32 计算：

$$
650332\times2\times4\approx5.2\ \mathrm{MB}.
$$

Replay 主实验的实际序列化 buffer 同样限制为约 5.2 MB，容许误差不超过 10%。

同时报告：

- EWC 额外存储；
- Replay buffer 实际字节数；
- Replay 样本数量和任务分布；
- optimizer steps；
- samples processed；
- GPU-hours；
- 峰值显存；
- Fisher 计算时间。

## 6. 正式实验矩阵

### 6.1 P0：核心 continual 实验

| ID | 轨道 | 顺序 | 方法 |
|---|---|---|---|
| R1 | Realistic | 自然顺序 | SEQ |
| R2 | Realistic | 自然顺序 | Online EWC |
| R3 | Realistic | 联合训练 | Mix-total |
| S1 | Stress | 高冲突顺序 | SEQ |
| S2 | Stress | 高冲突顺序 | Online EWC |
| S3 | Stress | 高冲突顺序 | Replay |
| S4 | Stress | 联合训练 | Mix-total |

### 6.2 P0：Stress 单任务 Oracle

每个 Oracle 都从同一个 A101-T0 权重出发：

| ID | 训练数据 |
|---|---|
| O1 | 仅 Stress-T1 |
| O2 | 仅 Stress-T2 |
| O3 | 仅 Stress-T3 |
| O4 | 仅 Stress-T4 |

四个 Oracle 各训练一个任务阶段，总计算量约等于一个完整 T1–T4 sequential run。

### 6.3 P1：顺序敏感性

| ID | 轨道 | 顺序 | 方法 |
|---|---|---|---|
| S5 | Stress | 反向高冲突顺序 | SEQ |
| S6 | Stress | 反向高冲突顺序 | Online EWC |

### 6.4 Stretch goals

按以下顺序执行：

1. Stress Mix-exposure；
2. Realistic Replay；
3. Realistic T1–T4 单任务 Oracle；
4. Realistic Mix-exposure。

若训练发生失败，优先保留重跑预算，不启动 stretch goals。

## 7. Pilot 与冻结规则

### 7.1 训练步数

SEQ 测试：

$$
1\mathrm{k},\quad5\mathrm{k},\quad20\mathrm{k}\ \text{steps/task}.
$$

选择满足以下条件的最低预算：

- 当前任务基本收敛；
- 增加步数后的当前任务收益已经较小；
- 历史任务退化可以测量；
- 完整 sequential run 目标不超过约 2.4 GPU-days。

不能无限增加步数以人为制造遗忘。

### 7.2 EWC $\lambda$

在第一次任务切换时测量：

$$
\frac{\lambda L_{\mathrm{EWC}}}{L_{\mathrm{data}}}.
$$

测试该比值约为：

$$
0.1\%,\quad1\%,\quad10\%,\quad100\%.
$$

根据 stability–plasticity trade-off 选择 $\lambda$，并在所有正式 EWC run 前冻结。

### 7.3 Replay 比例

在 5.2 MB 固定 buffer 下短测：

$$
10\%,\quad25\%,\quad50\%.
$$

根据当前任务 LER、历史任务 LER 和五任务宏平均 LER 选择一个比例，正式 Replay run 前冻结。

### 7.4 冻结清单

正式训练开始前，必须在 experiment manifest 中冻结：

- Realistic/Stress 完整噪声参数；
- 两个 Stress 任务顺序；
- continual training seed；
- 每任务 optimizer steps；
- EWC $\lambda$ 和 $\gamma$；
- Fisher 样本数与 estimator；
- Replay 比例、buffer 字节上限和选样规则；
- validation/test seeds 与 shots；
- 成功判据。

## 8. Checkpoint 与评测矩阵

令实际任务顺序为 $\pi=(\pi_1,\pi_2,\pi_3,\pi_4)$，且 $\pi_0=T0$。

SEQ/EWC/Replay 保存：

$$
C_0=\text{A101-T0},\quad C_1,\quad C_2,\quad C_3,\quad C_4,
$$

其中 $C_i$ 表示完成实际顺序中第 $i$ 个任务后的固定 endpoint。

每个 checkpoint 都评测所有 T0–T4：

$$
R_{i,j}=LER(C_i,T_j).
$$

由此形成完整 `5×5` 矩阵，包括尚未训练任务的上三角。Stress 非自然顺序下，行表示实际训练阶段，列始终固定为 T0、T1、T2、T3、T4。

Mix 在完成 0%、25%、50%、75%、100% 总更新量时保存 checkpoint，但其曲线只表示联合优化过程，不称为 sequential forgetting。

禁止使用跨任务全局 `best_model` 作为正式 endpoint。best-current-task checkpoint 仅用于诊断。

## 9. 指标体系

标准 continual-learning 指标采用：

$$
S_{i,j}=1-LER_{i,j},
$$

使指标保持“越高越好”的标准方向。原始 LER 矩阵必须同时报告。

完整评测矩阵、ACC/BWT/FWT 参考 GEM 评测框架；forgetting 与 intransigence 的联合评价参考 RWalk：

- Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning*, NeurIPS 2017: <https://arxiv.org/abs/1706.08840>
- Chaudhry et al., *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence*, ECCV 2018: <https://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf>

### 9.1 Final Average LER

$$
FALER=\frac{1}{5}\sum_{j=0}^{4}LER_{4,j}.
$$

T0–T4 全部纳入最终综合性能。

### 9.2 Average Incremental Performance

令 $H_i=\{T0,\pi_1,\ldots,\pi_i\}$ 为阶段 $i$ 已学习任务集合：

$$
AIP=
\frac{1}{5}\sum_{i=0}^{4}
\left(
\frac{1}{|H_i|}\sum_{T_j\in H_i}S_{i,j}
\right).
$$

该指标衡量整个训练过程，而不只看最终 checkpoint。

### 9.3 Task-wise Forgetting

令 $a(j)$ 表示任务 $T_j$ 的学习阶段。仅对 $a(j)<4$ 的历史任务定义：

$$
f_j=
\max_{\ell\in\{a(j),\ldots,3\}}S_{\ell,j}-S_{4,j}.
$$

$$
AF=\frac{1}{|H_3|}\sum_{T_j\in H_3}f_j.
$$

解释：

- $f_j>0$：遗忘；
- $f_j=0$：保持；
- $f_j<0$：正后向迁移。

最后学习的任务没有经历后续任务，因此该顺序中的 forgetting 为 `N/A`，不能人为记为零。T4 会在 Stress 反向顺序中接受遗忘评测。

### 9.4 Backward Transfer

$$
BWT=
\frac{1}{|H_3|}
\sum_{T_j\in H_3}
\left(S_{4,j}-S_{a(j),j}\right).
$$

- $BWT<0$：后续学习损害旧任务；
- $BWT>0$：后续学习改善旧任务。

### 9.5 Forward Transfer

对于实际顺序中第 $i$ 个新任务，比较学习该任务前的性能与 Stage-0 性能：

$$
FWT_i=S_{i-1,\pi_i}-S_{0,\pi_i}.
$$

$i=1$ 时两者相同，固定为零，因此总体 FWT 对 $i=2,3,4$ 求平均：

$$
FWT=\frac{1}{3}\sum_{i=2}^{4}FWT_i.
$$

该指标用于判断“没有遗忘”是否伴随任务间正向迁移。

### 9.6 Acquisition Gap

对于 Stress 任务，令 $O_j$ 为从 A101-T0 独立训练任务 $T_j$ 的单任务 Oracle：

$$
AG_j=LER(C_{a(j)},T_j)-LER(O_j,T_j).
$$

- $AG_j\approx0$：新任务学习充分；
- $AG_j>0$：持续学习约束阻碍了新任务适应；
- $AG_j<0$：历史任务产生正向迁移。

该指标是适用于当前 domain-incremental 解码任务的 intransigence proxy，不冒充原论文中的严格 class-incremental intransigence。

### 9.7 解码与工程指标

同时报告：

- BCE/NLL；
- residual syndrome density；
- PyMatching 后端延迟；
- Fisher-weighted parameter drift；
- 分层参数漂移；
- GPU-hours；
- 峰值显存；
- 额外存储；
- samples processed。

不将全部指标加权成单一总分。

## 10. 成功判据

### 10.1 SEQ 确实发生遗忘

同时检查：

- AF 为正；
- 历史任务绝对 $\Delta LER$ 超过配对测试误差；
- 至少多个历史任务出现一致退化；
- 相对 5% 作为效应量参考，不作为唯一判据。

### 10.2 EWC 有效

应同时满足：

1. AF 低于 SEQ；
2. BWT 比 SEQ 更接近零或转为正；
3. Final Average LER 不明显恶化；
4. Acquisition Gap 不超过预注册容忍值；
5. 当前任务 LER 不比 SEQ 相对恶化超过 5%。

`AF 降低 50%` 保留为强效果目标，不作为唯一成功条件。

### 10.3 EWC 与 Replay

在约 5.2 MB 等存储条件下比较：

- AF；
- BWT；
- Acquisition Gap；
- Final Average LER；
- GPU-hours。

若 Replay 明显优于 EWC，应如实结论为简单样本记忆在该任务上更有效。

## 11. 单 seed 与推理统计

- 正式训练前固定一个新的 continual seed；
- 不因结果不理想而更换 seed；
- 所有方法共享对应任务的数据随机流；
- validation/test seed 与训练 seed 分离；
- 所有 checkpoint 使用配对测试数据；
- 使用配对 bootstrap 或配对逻辑错误差值区间；
- 增加 shots 只能降低推理测量误差，不能代替多训练 seed；
- 不使用“统计显著优于”或“跨 seed 稳定”的表述。

## 12. OOD 与 Willow

仅当以下条件成立时，才对最终 SEQ、EWC、Replay、Mix 执行额外 OOD 推理：

- Stress-SEQ 确实发生遗忘；
- EWC 降低 AF；
- EWC Acquisition Gap 可接受；
- EWC 当前任务性能没有严重下降。

额外评测包括：

- d=7/d=9 的 110 个 OOD 配置；
- Willow d=5/d=7 零微调；
- residual syndrome density；
- PyMatching 后端延迟。

OOD 结果不用于证明 EWC 本身有效。

## 13. 实施前必须修复

1. 所有任务配置显式固定 A101 开关；
2. 建立独立 SEQ launcher；
3. 修正并验证 Fisher estimator；
4. Fisher 与训练 loss weighting 对齐；
5. 每个 run 隔离 Fisher 状态；
6. mix 配置通过公共 validator；
7. mix validation 覆盖 T0–T4；
8. 增加 stage×task matrix evaluator；
9. 禁止跨任务 `best_model` 作为正式 endpoint；
10. 增加 manifest、配置哈希和 checkpoint 哈希记录。

## 14. 两周资源预算

资源上限：

- 2 张 GPU；
- 14 天；
- 约 28 GPU-days。

预计预算：

| 项目 | GPU-days |
|---|---:|
| Pilot、任务筛选和校准 | 约 2.8 |
| P0 七个正式 continual run | 约 16.8 |
| Stress 四个单任务 Oracle | 约 2.4 |
| P1 两个反向顺序 run | 约 4.8 |
| 合计 | 约 26.8 |
| 预留 | 约 1.2 |

执行优先级：

1. Pilot 和实现验证；
2. Stress SEQ/EWC/Replay/Mix-total；
3. Stress 单任务 Oracle；
4. Realistic SEQ/EWC/Mix-total；
5. Stress 反向 SEQ/EWC；
6. Stretch goals。

Replay 不因时间不足优先删除。若单个 run 接近三天上限，则先推迟 P1，不牺牲 P0 的因果对照。

## 15. 交付物

实验目录最终应包含：

- `qewc_v3_experiment_design.md`；
- `experiment_manifest.yaml`；
- `tasks_realistic.yaml`；
- `tasks_stress.yaml`；
- `run_matrix.yaml`；
- `metrics_spec.md`；
- stage checkpoint 清单；
- checkpoint/config SHA256；
- Fisher 统计；
- Replay buffer 清单；
- 配对测试数据 seed/hash；
- 原始 `5×5` JSON/CSV；
- FALER/AF/BWT/FWT/AIP/AG 汇总表；
- LER 热图；
- forgetting 曲线；
- stability–plasticity 图；
- EWC–Replay 等存储对比；
- 失败、中断和重跑记录。

## 16. 结果解释规则

- **Realistic SEQ 不遗忘**：现实噪声任务之间干扰较弱，EWC 在该轨道未显示必要性；
- **Stress SEQ 不遗忘**：任务冲突或训练预算仍不足，不能声称 EWC 成功；
- **EWC AF 更低但 AG 更高**：EWC 通过牺牲可塑性换取稳定性；
- **EWC 同时降低 AF 且保持 AG**：支持 EWC 有效缓解遗忘；
- **Replay 优于等存储 EWC**：该任务更适合有限样本记忆；
- **BWT/FWT 为正**：任务之间存在有益迁移，应与“没有遗忘”分开表述；
- **EWC 接近 Mix-total**：支持无历史数据条件下接近联合训练上界的结论。

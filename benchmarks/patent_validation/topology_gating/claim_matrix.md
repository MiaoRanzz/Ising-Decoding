# v2 技术点—证据矩阵

| v2 技术点 | 当前实现位置 | 当前证据 | 完成状态 |
|---|---|---|---|
| 类型化候选及数据/测量阈值 | `evaluation/topology_gating_v2.py::typed_candidates` | 单元测试 | 已实现 |
| 作用域重叠、邻接及冲突成簇 | `interaction_clusters` | 单元测试 | 已实现 |
| 簇是联合决策域而非整体提交集合 | `legal_combinations`、`gate_actions` | 子集优于整簇受控用例 | 已实现 |
| 空组合、互斥约束和预算搜索 | `legal_combinations` | 精确枚举及冲突测试 | 已实现 |
| 逐组合 GF(2) 残余和逻辑帧 | `evaluate_cluster`、`apply_actions` | 线性映射测试 | 已实现 |
| 工作量、不确定性和逻辑风险效用 | `workload_features`、`evaluate_cluster` | syndrome 退化但逻辑类别不同的测试 | 已实现 |
| 每簇至多一个组合 | `gate_actions` | 决策轨迹及测试 | 已实现 |
| 接纳后更新状态并重算 | `gate_actions` | 后续决策的 workload-before 变化测试 | 已实现 |
| 残余后端解码及局部逻辑帧组合 | Phase 1 benchmark | PyMatching 成对 smoke/validation | 已接入代理动作 |
| 四类真实空间动作的扩展 H/L | 待建 surface adapter | 尚无 | 待 Phase 2 |
| 真实预训练 Ising 四通道概率 | 待提供/训练 checkpoint | 本工作区无可用 checkpoint | 待 Phase 2 |
| 多 seed、多码距正式统计 | 冻结协议 | 当前配置仅单 seed | 待正式运行 |
| 优化后端到端时延 | 待 CUDA/C++ 路径 | Python 参考耗时不可外推 | 不在 Phase 1 结论内 |

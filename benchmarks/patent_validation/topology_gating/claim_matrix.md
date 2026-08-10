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
| 四类真实空间动作的扩展 H/L | `evaluation/surface_topology_adapter.py` | 方程一致性、边界及动作位置测试 | 已实现 |
| 真实 Ising 四通道概率与拓扑联合损失 | `topology_gating_v2_small.py`、`training/topology_loss.py` | 真实模型 smoke；正式 3-seed 结果待运行 | 已接入，待增强型运行 |
| 多 seed、多码距正式统计 | 增强型配置已冻结 3 seed、单码距 | 4 卡入口和哈希安全续跑测试 | 单码距待运行；第二码距仍缺失 |
| 优化后端到端时延 | 待 CUDA/C++ 路径 | Python 参考耗时不可外推 | 不在 Phase 1 结论内 |

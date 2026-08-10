# 拓扑门控 v2 证据索引

## 受版本控制的证据

- 验证协议：`benchmarks/patent_validation/topology_gating/protocol.md`
- 技术点矩阵：`benchmarks/patent_validation/topology_gating/claim_matrix.md`
- 冻结配置：`conf/experiments/patent/topology_gating_v2.yaml`
- 门控内核：`code/evaluation/topology_gating_v2.py`
- Phase 1 入口：`code/benchmarks/patent_validation/topology_gating_v2.py`
- 正确性测试：`code/tests/test_topology_gating_v2.py`

## 本地生成证据

- `outputs/patent_validation/topology_gating_v2/smoke/manifest.json`
- `outputs/patent_validation/topology_gating_v2/smoke/summary.csv`
- `outputs/patent_validation/topology_gating_v2/smoke/results.md`
- `outputs/patent_validation/topology_gating_v2/pilot/`（单 seed 初步冻结结果）
- `outputs/patent_validation/topology_gating_v2/validation/`（单 seed 大样本运行后生成，
  尚需至少 3 个独立 seed 汇总后才能作为正式证据）

本索引不复制原始输出；manifest 中的配置哈希、commit 和环境信息用于将外部归档结果
绑定到代码版本。正式归档时还应记录整个结果目录的 SHA-256 清单和持久化存储位置。

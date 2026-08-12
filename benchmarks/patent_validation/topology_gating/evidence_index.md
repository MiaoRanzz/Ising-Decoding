# 拓扑门控 v2 证据索引

## 受版本控制的证据

- 验证协议：`benchmarks/patent_validation/topology_gating/protocol.md`
- 技术点矩阵：`benchmarks/patent_validation/topology_gating/claim_matrix.md`
- 冻结配置：`conf/experiments/patent/topology_gating_v2.yaml`
- 门控内核：`code/evaluation/topology_gating_v2.py`
- Phase 1 入口：`code/benchmarks/patent_validation/topology_gating_v2.py`
- 正确性测试：`code/tests/test_topology_gating_v2.py`
- Phase 2 四通道适配：`code/evaluation/surface_topology_adapter.py`
- 拓扑联合损失：`code/training/topology_loss.py`
- 4 卡增强型入口：`code/benchmarks/patent_validation/topology_gating_v2_small.py`
- 一键启动：`code/scripts/patent/run_topology_gating_v2_small.sh`
- 增强型冻结配置：`conf/experiments/patent/topology_gating_v2_small.yaml`
- Phase 2 正确性测试：`code/tests/test_surface_topology_adapter.py`、`code/tests/test_topology_gating_v2_small.py`

## 本地生成证据

- `outputs/patent_validation/topology_gating_v2/smoke/manifest.json`
- `outputs/patent_validation/topology_gating_v2/smoke/summary.csv`
- `outputs/patent_validation/topology_gating_v2/smoke/results.md`
- `outputs/patent_validation/topology_gating_v2/pilot/`（单 seed 初步冻结结果）
- `outputs/patent_validation/topology_gating_v2/validation/`（单 seed 大样本运行后生成，
  尚需至少 3 个独立 seed 汇总后才能作为正式证据）

- `outputs/patent_validation/topology_gating_v2_small/smoke/`（128/64/256 链路检查，数值不作效果证据）
- `outputs/patent_validation/topology_gating_v2_small/full/`（4 卡 8 小时单码距增强型 pilot，仍非正式申请证据）

本索引不复制原始输出；manifest 中的配置哈希、commit 和环境信息用于将外部归档结果
绑定到代码版本。正式归档时还应记录整个结果目录的 SHA-256 清单和持久化存储位置。

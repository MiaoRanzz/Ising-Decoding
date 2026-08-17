# 拓扑残差效用经典门控

这里实现交底书技术方案中的模型后经典处理。核心流程为：

1. 将四通道 Ising-fast logit 校准为候选动作概率；
2. 根据扩展 detector 映射 `H~` 的作用域重叠和局部拓扑邻接成簇；
3. 在有界簇内枚举/排序合法动作组合，空组合始终隐式存在；
4. 对每个组合计算试探残余 `s' = s xor H~e` 和逻辑帧增量 `l = L~e`；
5. 计算残余工作量、不确定性、逻辑风险代理量和门控开销；
6. 接纳当前效用最高且满足硬约束的组合，更新状态后重新成簇；
7. 将最终残余 syndrome 交给 PyMatching，并把 `L~e` 与全局预测组合。

`classical_gate.py` 是纯 NumPy 的通用门控，不读取真实 observable。
`surface_code.py` 通过仓库当前 `PreDecoderMemoryEvalModule` 的单位动作响应，
离线生成与生产后处理完全同序的 `H~`、`L~`。其中 `L~e` 只是已提交动作的
逻辑帧增量，不代表动作是否修复了未知的真实逻辑错误。

## Evaluation

先在 `settings.yaml` 中填写已有 paired corpus 和 checkpoint。当前仓库有示例
checkpoint，但没有提交体积较大的 corpus；如果配置的 `dataset_dir` 尚不存在，
先按 `my_file/end_to_end/L_logical/end_to_end.yaml` 生成：

```powershell
python my_file/end_to_end/L_logical/generate_labeled_dataset.py
```

门控 evaluation 不会使用其中的 `train_y`，复用 paired corpus 是为了取得严格同序的
`train_x`、detector 和仅供最终计分的 observable。随后从仓库根目录运行：

```powershell
python my_file/patent/evaluate.py
```

快速检查可覆盖样本数：

```powershell
python my_file/patent/evaluate.py --max-samples 256
```

JSON 报告在同一批 shot 上比较：

- 原始 syndrome 直接 PyMatching；
- Ising-fast 的全部阈值动作；
- 专利经典门控筛选后的动作。

报告包含 LER、残余权重、动作数、paired helpful/harmful 计数、组合搜索量和
门控耗时。observable 仅在全局解码完成后用于统计 LER，不会进入模型或门控。

## Tests

```powershell
python -m unittest my_file.patent.test_classical_gate
```

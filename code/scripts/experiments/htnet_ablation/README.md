# HTNet T0 three-factor ablation

This experiment keeps HTNet depth, width, adaptive branch fusion, activation,
dropout, optimizer, and training data fixed. It toggles only:

- branch mixer;
- AxisChannelGate versus ChannelGate;
- raw-input concatenation in the output head.

The eight cell names encode these switches in that order. For example, `a101`
enables the branch mixer and input-skip head but uses ChannelGate.

## Training

Run two independent single-GPU queues under nohup:

```bash
cd /root/project/Ising-Decoding
nohup env PREDECODER_PYTHON=/root/miniconda3/envs/ising-decoding/bin/python \
  bash code/scripts/experiments/htnet_ablation/run_dual_a100.sh \
  > logs/htnet_ablation_t0_dual_a100_nohup.log 2>&1 &
echo $!
```

GPU 0 runs `a000,a011,a101,a110`; GPU 1 runs
`a001,a010,a100,a111`. Each cell trains from scratch for 100 epochs with
4,194,304 samples per epoch and seed 12345. The script refuses to start a fresh
cell over an existing models directory.

## Paired evaluation

After all training processes finish, run the evaluation without concurrent
training jobs:

```bash
cd /root/project/Ising-Decoding
nohup env CUDA_VISIBLE_DEVICES=0 \
  PREDECODER_PYTHON=/root/miniconda3/envs/ising-decoding/bin/python \
  bash code/scripts/experiments/htnet_ablation/run_paired_evaluation.sh \
  > logs/htnet_ablation_t0_paired_eval_nohup.log 2>&1 &
echo $!
```

The evaluation uses 1,048,576 paired shots per X/Z basis. Residual PyMatching
latency is warmed up and measured five times; the reported primary latency is
the median. It excludes neural-network forward latency.

Outputs:

- `outputs/htnet_ablation/htnet_t0_three_factor_s1048576_seed12345.json`
- the same stem with `.csv` for model-level metrics;
- the same stem with `_effects.csv` for LER main effects, interactions, and Holm
  adjusted p-values;
- the same stem with `_latency_effects.csv` for residual PyMatching latency
  main effects and interactions;

A negative `effect_ler` means enabling the component lowers LER. Formal
component claims use the three combined-basis main effects after Holm
correction. Interaction rows are exploratory.

Only one training seed is used. Test-shot confidence intervals do not measure
training-seed variability.

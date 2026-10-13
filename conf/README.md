# Configuration layout

Hydra config names are paths relative to `conf/` without the `.yaml` suffix.

## Stable public configs

- `config_public.yaml` is the primary public entry point.
- `presets/surface/`, `presets/color/`, and `presets/development/` contain
  reusable public presets.

For example: `CONFIG_NAME=presets/surface/config_qec_decoder_r9_fp8`.

## Two-model release examples

- `examples/qadapt/config_qadapt_t0_base.yaml` through
  `config_qadapt_t4_z_bias_1p5.yaml`: one shared five-task sequence for
  QAdapt sequential+EWC training and simulated inference.
- Ising-Fast reuses `config_qadapt_t0_base.yaml` with the public
  `model_id=1` override and trains from random initialization for 100 epochs.
- The runnable training, final-weight export, T0--T4, OOD, and Willow entry
  points are documented in `code/examples/README.md`.

## Research experiments

- `experiments/external_qpu/` contains hardware/external-sample comparison
  configs.
- `experiments/google_qec/` contains Google QEC learned-noise training configs.
- `experiments/google_qec/noise_models/` stores fitted and network-predicted
  noise-model snapshots used by those experiments.

Use the full Hydra name for new commands, for example
`experiments/google_qec/config_google_d3_d5_r13_noise_network`. Historical
unique basenames remain supported by `scripts.config_paths.config_path`.

Generated OOD configurations belong under `outputs/generated_configs/`, not
under `conf/`. The `.gitignore` allowlist intentionally keeps only
`config_public.yaml`, `presets/**/*.yaml`, the listed QAdapt task configs, and
`experiments/**/*.yaml` in source control.

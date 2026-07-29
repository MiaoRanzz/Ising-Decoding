# Configuration layout

Hydra config names are paths relative to `conf/` without the `.yaml` suffix.

## Stable public configs

- `config_public.yaml` is the primary public entry point.
- `presets/surface/`, `presets/color/`, and `presets/development/` contain
  reusable public presets.

For example: `CONFIG_NAME=presets/surface/config_qec_decoder_r9_fp8`.

## QAdapt examples

- `examples/qadapt/config_qadapt_t0_base.yaml` through
  `config_qadapt_t4_z_bias_1p5.yaml`: one shared five-task sequence for
  sequential+EWC training and simulated inference.

Generated OOD configurations belong under `outputs/generated_configs/`, not
under `conf/`. The `.gitignore` allowlist intentionally keeps only
`config_public.yaml`, `presets/**/*.yaml`, and the five listed QAdapt task
configs in source control.

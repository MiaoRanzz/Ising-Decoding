# Standalone Ising surface decoder

This directory contains a self-contained deployment core for the trained
surface-code pre-decoder and PyMatching backend. It is fixed to:

- rotated surface code `d=9`, `T=9`, orientation `XV`;
- HTnet public model ID 111, checkpoint epoch 82;
- batch-one detector input in Stim emission order;
- deterministic threshold inference in FP32;
- one basis-specific decoder instance for X and one for Z.

It does not import training, evaluation, data, workflow, or model modules from
the parent research repository.

## Input contract

`SurfaceDecoderCore.decode_detectors` accepts one contiguous NumPy array with
shape `(720,)`, dtype `uint8`, and values restricted to 0 or 1. This is a
detector-event vector, not raw per-round stabilizer measurements.

## Installation

Use a Python 3.11+ environment with a PyTorch build matching the target CUDA
runtime. From this directory:

```bash
pip install -e .
```

## Python API

```python
import numpy as np

from ising_decoder import DecoderConfig, SurfaceDecoderCore

config = DecoderConfig(
    checkpoint_path="artifacts/HTnet.0.82.pt",
    geometry_path="artifacts/geometry_d9_xv.npz",
    dem_path="artifacts/surface_d9_t9_xv_t0_X.dem",
    basis="X",
    device="cuda",
)
decoder = SurfaceDecoderCore(config)
result = decoder.decode_detectors(np.zeros(720, dtype=np.uint8))
print(result.logical_flip)
```

## Replay and benchmark

The CLI reads Stim sparse `.dets` files containing 720 detectors and one
observable per shot:

```bash
ising-decode replay tests/fixtures/golden_X.dets basis=X device=cuda
ising-decode benchmark tests/fixtures/golden_X.dets basis=X device=cuda
```

Hydra overrides can select the basis, device, artifact paths, sample count, and
report output path. For example:

```bash
ising-decode benchmark tests/fixtures/golden_Z.dets \
  basis=Z device=cpu benchmark.samples=20 output_path=benchmark.json
```

## Artifact policy

The runtime checks that configured files exist. It loads the checkpoint with
`torch.load(..., weights_only=True)` and applies the state dict strictly, but it
does not enforce a manifest or SHA-256 policy. The bundled T0 DEM files are
reference artifacts; a real QPU deployment must supply basis-specific DEMs
whose detector ordering matches the incoming detector vector.

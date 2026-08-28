# Surface decoder benchmark

Run the package benchmark from the deployment directory:

```bash
ising-decode benchmark tests/fixtures/golden_X.dets basis=X device=cuda
```

The reported hot-path latency starts with a ready 720-bit detector vector and
ends after HTnet, device-to-host residual transfer, PyMatching, and logical XOR.

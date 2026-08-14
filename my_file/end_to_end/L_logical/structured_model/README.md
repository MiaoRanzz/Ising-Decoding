# Integrated structured-action Ising-fast experiment

This directory replaces the post-hoc safe-no-op gate with one model whose
output action space explicitly includes no-op:

```text
data packet: {no-op, Z, X, Y}
X syndrome:  {no-op, apply}
Z syndrome:  {no-op, apply}
```

The first model is hot-started from the configured Ising-fast checkpoint. Its
initial structured logits reproduce the original four independent thresholded
actions apart from exact threshold ties. The workflow then trains on `trainY`, generates endpoint
teachers through the real `predecoder + PyMatching` path, and fine-tunes the
same model with both losses.

Edit `settings.yaml`, then run from repository root. There are no command-line
configuration switches. Initially leave `structured_training.phase: oracle`:

```bash
python my_file/end_to_end/L_logical/structured_model/train_structured_ising.py
python my_file/end_to_end/L_logical/structured_model/generate_endpoint_teacher.py
```

Then change these YAML sections:

```yaml
structured_training:
  phase: teacher

structured_teacher_training:
  resume_checkpoint: my_file/end_to_end/L_logical/structured_model/models/oracle/best.pt
```

and run:

```bash
python my_file/end_to_end/L_logical/structured_model/train_structured_ising.py
python my_file/end_to_end/L_logical/structured_model/evaluate_structured_ising.py
```

For a second teacher round, point `structured_teacher_generation.checkpoint`
at the preceding teacher checkpoint, use a new output directory, update
`structured_teacher_training.teacher_dataset_dir`, and resume from that same
checkpoint. The evaluation script chooses its no-op bias solely on the fixed
validation split and reports the untouched test split once.

`logical_failure_weight` is deliberately dominant. Residual weight and action
count are only tie breakers, so the teacher does not learn to minimize residual
at the expense of logical correctness.

Both training phases report progress during execution. Set
`log_every_batches` in the corresponding training YAML section to choose the
reporting interval; each line includes current loss, oracle/endpoint loss,
learning rate, throughput, and estimated time remaining in the epoch.

Evaluation reports four paths on exactly the same held-out shots: raw
PyMatching, the original Ising-fast checkpoint, the structured oracle model,
and the structured teacher model. `oracle_checkpoint` and `checkpoint` in
`structured_evaluation` select the latter two paths.

The evaluation JSON also contains `warm_start_audit`. It compares the original
four-logit actions to an untrained structured model freshly converted from the
same Ising-fast checkpoint. Zero action and endpoint mismatches prove that any
oracle-stage LER regression was caused by structured-CE fine-tuning, not by
the head conversion.

After changing conversion code, first set `structured_evaluation.mode` to
`warm_start_audit`. This mode deliberately ignores old structured checkpoints,
which may have an incompatible trunk layout. Once the action, fixed-action
failure, and fixed-action residual mismatch counts are zero (apart from any
explicitly diagnosed exact-threshold ties), retrain oracle and teacher from
scratch and change the mode to `full`.
Both original and structured action tensors are evaluated through the same
fixed-action endpoint. The original model's complete pipeline is reported only
as a precision/preprocessing reference and is not used for the equivalence
mismatch count.

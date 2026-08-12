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

Then change this YAML section:

```yaml
structured_training:
  phase: teacher
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

# Labeled end-to-end evaluation

This directory contains a paired-data workflow for testing three decoders on
the *same* shots:

1. raw detectors -> PyMatching;
2. raw detectors -> an Ising-fast checkpoint -> PyMatching;
3. raw detectors -> the paired `trainY` oracle action -> PyMatching.

`generate_labeled_dataset.py` writes a corpus whose detector/observable rows
and `trainX`/`trainY` rows have the same row index.  The label is therefore
available only for synthetic Torch-generator data; it cannot be reconstructed
from a generic `.dets` file or hardware shot.

`compare_three_paths.py` reads that corpus and writes a JSON/NPZ report.  The
oracle route deliberately runs through `PreDecoderMemoryEvalModule`, matching
the production residual and `pre_L` construction rather than assuming a zero
residual by hand.

Both scripts read `end_to_end.yaml` beside them by default.  Edit that file to
set paths, checkpoint, batch size, and sample count; `distance`, `n_rounds`,
and rotation default to the referenced project YAML.  Once configured, run
each script without repeating its parameters:

```powershell
python my_file/end_to_end/generate_labeled_dataset.py
python my_file/end_to_end/compare_three_paths.py
```

Use `--settings path/to/other.yaml` for a separate experiment, or pass any
ordinary option (for example `--num-samples 4096`) as a one-off override.

## Whole-shot safe no-op analysis

After `compare_three_paths.py` has produced its `.per_shot.npz` file, edit the
`safe_no_op` section in `end_to_end.yaml` and run:

```powershell
python my_file/end_to_end/L_logical/analyze_safe_no_op.py
```

It stores compact shot-level logit confidence summaries and selects a whole-shot
no-op threshold on one random split before reporting its LER on the held-out
split.  `positive_margin_q10` is the default score: the 10th percentile of
`|logit|` among the corrections the model proposed for that shot.

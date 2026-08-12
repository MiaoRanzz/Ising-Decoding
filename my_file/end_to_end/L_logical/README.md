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
python my_file/end_to_end/L_logical/generate_labeled_dataset.py
python my_file/end_to_end/L_logical/compare_three_paths.py
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

## Packet-level safe no-op gate

The local safe no-op experiment is a second-stage model.  It freezes the
current Ising-fast proposal and learns whether to keep each proposed local
packet before PyMatching.  The data packet couples the two data-correction
channels at one spacetime site; the two syndrome channels each form one packet.
It therefore cannot accidentally retain half of a Y-like data correction.

All normal settings live in `end_to_end.yaml`; update the dataset/checkpoint
paths there, then run these commands from the repository root:

```bash
python my_file/end_to_end/L_logical/packet_model/generate_local_risk_dataset.py
python my_file/end_to_end/L_logical/packet_model/train_local_safe_no_op.py
python my_file/end_to_end/L_logical/packet_model/evaluate_local_safe_no_op.py
```

The first command is intentionally expensive: for each proposal packet it
compares final `predecoder + PyMatching` failure with and without that packet.
For an initial check, set `packet_risk_generation.num_samples: 4096`; use a new
empty `output_dir` for a full run.  It writes only proposal logits and packet
effects and refers back to the existing source corpus for `train_x`, so it does
not duplicate the large input tensor.

`packet_effect.npy` holds `+1` helpful, `-1` harmful, `0` neutral and `-2`
not proposed. Training keeps every helpful/harmful label but randomly
downsamples neutral loss positions. It writes one `epoch_*.pt` candidate per
configured interval. The final command selects the candidate checkpoint and
gate threshold by final LER on the validation split, then reports PyMatching,
frozen proposal + PyMatching, and local-gate + PyMatching once on a separate
test split.

## Group-level safe no-op gate

`group_model/` is a separate experiment: it first groups neighbouring *active*
proposal packets using the fixed `group_risk_generation.grouping` rules, then
learns one harmful-risk logit per group.  A low-risk or uncertain group is
retained; only a high harmful-risk logit can turn it into a no-op.  A group's
counterfactual label is
computed by removing all of its packets together and measuring final LER.
This explicitly captures interactions that the packet model's one-packet
counterfactual misses.

```bash
python my_file/end_to_end/L_logical/group_model/generate_local_group_risk_dataset.py
python my_file/end_to_end/L_logical/group_model/train_local_group_safe_no_op.py
python my_file/end_to_end/L_logical/group_model/evaluate_local_group_safe_no_op.py
```

After selecting a checkpoint and veto threshold on validation, copy those two
values into `group_gate_fixed_test` and run exactly one held-out test without
rescanning alternatives:

```bash
python my_file/end_to_end/L_logical/group_model/test_fixed_group_gate.py
```

The group data is variable-length: `shot_group_ptr.npy` assigns consecutive
groups to each shot, `group_member_ptr.npy` assigns members to each group, and
`group_members.npy` stores `[packet_type, round, row, column]`.  Group labels
are stored in `group_effect.npy` as `+1` helpful, `0` neutral, or `-1` harmful.
The packet and group YAML sections have no configuration fallback between them.
The group evaluator always includes an accept-all candidate (the frozen
proposal) and an all-no-op candidate (raw PyMatching); learned thresholds are
restricted by `group_gate_evaluation.max_veto_coverage`.
The fixed test checks that the explicitly configured Ising-fast checkpoint is
the one that generated the risk dataset, then reports harmful-label lift,
harmful recall, and helpful false-veto rate for the selected veto set.

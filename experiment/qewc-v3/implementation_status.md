# QEWC v3 implementation status

Updated: 2026-09-07 UTC.

| Design prerequisite | Status | Evidence / remaining work |
|---|---|---|
| Explicit A101 switches and frozen noise configs | Complete | Realistic and Stress configs use d9/r9/both/O1, A101 toggles, and skip-upscaling; source tests verify hashes and task deltas. |
| Independent SEQ entry | Complete | `run_seq_pilot.sh` and `run_continual.py --method seq` force EWC/Replay off and use weights-only stage transitions. |
| True empirical diagonal Fisher | Complete | Per-sample BCE gradients use `torch.func.vmap`; exact numeric test passes; 4,096 vs 65,536 cosine is 0.99909. |
| Online EWC isolation and diagnostics | Complete | Run-local Fisher states, gamma update, fail-fast state checks, penalty ratio logging, per-layer Fisher/drift JSON, and Fisher timing are wired and tested. |
| Replay fixed-memory semantics | Complete | 594-sample deterministic task/X-Z-balanced buffer serializes to 5,229,158 bytes; fixed total batch and sample-weighted 10/25/50 percent replacement are tested. |
| Mix-total training and validation | Complete | Equal-weight round-robin batches, five-task validation, macro checkpoint metric, Realistic/Stress Mix configs, and launcher are validated. |
| Stress screening and orders | Complete | 16-candidate LER plus fixed-batch gradient/representation scan is complete; tasks and high-conflict/reverse orders are frozen. |
| Common paired matrix evaluator and metrics | Complete | Explicit T0-T4 configs, arbitrary C0-C4 checkpoints, paired CIs, and AF/BWT/FWT/AG summaries are implemented and tested. |
| Resource/provenance capture | Complete | Endpoint/config hashes, optimizer steps, processed samples, GPU-hours, peak memory, Fisher time, and extra-storage bytes are emitted. |
| Step pilot | Complete and frozen | 1k/5k/20k were compared on identical 65,536-per-basis streams; neither larger budget improved T1 beyond its paired 95% interval; selected 1,000 steps. |
| EWC lambda pilot | Complete and frozen | Deterministic re-evaluation selected lambda 43.87539484535376; it pointwise dominates the other candidates on T0/T1. |
| Replay ratio pilot | Complete and frozen | 10/25/50 percent used the same 594-sample memory budget; selected 10 percent, while 25/50 percent failed the T1 plasticity gate. |
| Immutable evaluation streams | Complete and frozen | Realistic and Stress each contain fixed 65,536-shot X/Z streams per task with manifest SHA256 and strict readback checks. |
| Formal reporting artifacts | Complete | Raw matrix JSON/CSV, metrics summaries, LER heatmaps, forgetting curves, and pilot stability-plasticity outputs are automated and tested. |
| Formal matrix | In progress | S1 (SEQ, GPU 0) and S2 (online EWC, GPU 1) started on the frozen high-conflict Stress order. |

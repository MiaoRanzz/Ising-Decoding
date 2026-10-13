# QEWC v3 metric contract

This file is an execution-facing summary of `qewc_v3_experiment_design.md`.
The design document remains the normative source.

- Matrix columns are always `T0,T1,T2,T3,T4`; sequential rows are `C0..C4` in actual training order.
- Persist raw LER and score `S = 1 - LER` as JSON and CSV.
- Report FALER, AIP, task-wise forgetting, AF, BWT, and FWT exactly as defined in sections 9.1–9.5.
- Stress runs additionally report acquisition gap against the four single-task oracles.
- The last task in an order has forgetting `N/A`, never zero.
- All checkpoint comparisons use paired samples and separate fixed validation/test streams.
- Report BCE/NLL from the acquisition-stage or five-task Mix validation streams; fixed `.dets` test files contain detector events and observables but no supervised correction targets, so they are not used to fabricate BCE.
- Do not select a cross-task global `best_model` as a stage endpoint.
- Report optimizer steps, processed samples, GPU-hours, peak memory, Fisher time, and serialized extra storage.

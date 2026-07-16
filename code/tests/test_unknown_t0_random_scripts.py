# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import NoiseModel, _single_p_mapping
from scripts.generate_unknown_t0_noise_configs import (
    generate_perturbed_noise_models,
    write_unknown_configs,
)
from scripts.run_unknown_t0_random_compare import (
    aggregate_rows,
    load_payloads,
    write_csv,
    write_markdown,
)


class TestUnknownT0NoiseConfigGeneration(unittest.TestCase):

    def _base_config_path(self, root: Path) -> Path:
        cfg = {
            "model_id": 111,
            "distance": 9,
            "n_rounds": 9,
            "workflow": {
                "task": "train"
            },
            "data": {
                "code_rotation": "O1",
                "noise_model": _single_p_mapping(0.001),
            },
        }
        path = root / "base_t0.yaml"
        OmegaConf.save(OmegaConf.create(cfg), path)
        return path

    def test_fixed_seed_generates_identical_config_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = self._base_config_path(root)
            out_a = root / "a"
            out_b = root / "b"

            paths_a, manifest_a = write_unknown_configs(
                base_config=base_config,
                output_dir=out_a,
                num_envs=3,
                frac=0.25,
                seed=20260714,
            )
            paths_b, manifest_b = write_unknown_configs(
                base_config=base_config,
                output_dir=out_b,
                num_envs=3,
                frac=0.25,
                seed=20260714,
            )

            self.assertEqual([p.name for p in paths_a], [p.name for p in paths_b])
            self.assertEqual(
                [p.read_text(encoding="utf-8") for p in paths_a],
                [p.read_text(encoding="utf-8") for p in paths_b],
            )
            self.assertEqual(manifest_a["seed"], manifest_b["seed"])
            self.assertEqual(manifest_a["environments"], manifest_b["environments"])

    def test_generated_noise_models_have_complete_valid_25p_mapping(self):
        base_noise = _single_p_mapping(0.001)

        generated = generate_perturbed_noise_models(base_noise, num_envs=5, frac=0.25, seed=20260714)

        self.assertEqual(len(generated), 5)
        for item in generated:
            noise = item["noise_model"]
            NoiseModel.from_config_dict(noise)
            self.assertEqual(set(noise), set(base_noise))
            for key, base_value in base_noise.items():
                self.assertGreaterEqual(noise[key], base_value * 0.75)
                self.assertLessEqual(noise[key], base_value * 1.25)


class TestUnknownT0RandomAggregation(unittest.TestCase):

    def _payload(self, env_name: str, noewc_ler: float, ewc_ler: float) -> dict:
        rows = []
        for basis in ("X", "Z"):
            rows.extend(
                [
                    {
                        "basis": basis,
                        "method": "pymatching",
                        "model_id": "",
                        "checkpoint": "",
                        "logical_errors": 100,
                        "samples": 1000,
                        "ler": 0.100,
                        "latency_us_per_round": 5.0,
                        "speedup_vs_pymatching": 1.0,
                    },
                    {
                        "basis": basis,
                        "method": "stfusion_domestic_e20",
                        "model_id": 111,
                        "checkpoint": "domestic20.pt",
                        "logical_errors": 90,
                        "samples": 1000,
                        "ler": 0.090,
                        "latency_us_per_round": 2.6,
                        "speedup_vs_pymatching": 1.9,
                    },
                    {
                        "basis": basis,
                        "method": "stfusion_domestic_e100",
                        "model_id": 111,
                        "checkpoint": "domestic100.pt",
                        "logical_errors": 80,
                        "samples": 1000,
                        "ler": 0.080,
                        "latency_us_per_round": 2.5,
                        "speedup_vs_pymatching": 2.0,
                    },
                    {
                        "basis": basis,
                        "method": "stfusion_mixed_e100",
                        "model_id": 111,
                        "checkpoint": "mixed100.pt",
                        "logical_errors": 85,
                        "samples": 1000,
                        "ler": 0.085,
                        "latency_us_per_round": 2.7,
                        "speedup_vs_pymatching": 1.8,
                    },
                    {
                        "basis": basis,
                        "method": "stfusion_seq_noewc_e100",
                        "model_id": 111,
                        "checkpoint": "seq-noewc100.pt",
                        "logical_errors": int(noewc_ler * 1000),
                        "samples": 1000,
                        "ler": noewc_ler,
                        "latency_us_per_round": 2.55,
                        "speedup_vs_pymatching": 1.95,
                    },
                    {
                        "basis": basis,
                        "method": "stfusion_seq_ewc_e100",
                        "model_id": 111,
                        "checkpoint": "seq-ewc100.pt",
                        "logical_errors": int(ewc_ler * 1000),
                        "samples": 1000,
                        "ler": ewc_ler,
                        "latency_us_per_round": 2.65,
                        "speedup_vs_pymatching": 1.85,
                    },
                ]
            )
        return {
            "config_name": env_name,
            "distance": 9,
            "n_rounds": 9,
            "num_samples": 1000,
            "latency_num_samples": 128,
            "seed": 12345,
            "device": "cpu",
            "rows": rows,
            "summary": [
                {
                    "method": method,
                    "ler_avg": sum(row["ler"] for row in rows if row["method"] == method) / 2,
                    "latency_us_per_round_avg": sum(
                        row["latency_us_per_round"] for row in rows if row["method"] == method
                    ) / 2,
                    "speedup_vs_pymatching_avg": sum(
                        row["speedup_vs_pymatching"] for row in rows if row["method"] == method
                    ) / 2,
                }
                for method in sorted({row["method"] for row in rows})
            ],
        }

    def test_aggregation_writes_summary_details_and_markdown_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "payloads"
            output_dir.mkdir()
            envs = ["config_unknown_t0_random_s20260714_e00", "config_unknown_t0_random_s20260714_e01"]
            (output_dir / "unknown_t0_random_e00.json").write_text(
                json.dumps(self._payload(envs[0], noewc_ler=0.070, ewc_ler=0.075)),
                encoding="utf-8",
            )
            (output_dir / "unknown_t0_random_e01.json").write_text(
                json.dumps(self._payload(envs[1], noewc_ler=0.060, ewc_ler=0.065)),
                encoding="utf-8",
            )

            payloads = load_payloads(output_dir, envs)
            detail_rows, summary_rows = aggregate_rows(payloads)
            summary_csv = root / "summary.csv"
            detail_csv = root / "details.csv"
            md_path = root / "comparison.md"
            write_csv(summary_csv, summary_rows)
            write_csv(detail_csv, detail_rows)
            write_markdown(
                md_path,
                summary_rows,
                envs,
                distance=9,
                n_rounds=9,
                num_samples=1000,
                latency_num_samples=128,
                seed=12345,
                basis="both",
                output_dir=output_dir,
                summary_csv=summary_csv,
                detail_csv=detail_csv,
            )

            summary = {row["method"]: row for row in csv.DictReader(summary_csv.open())}
            self.assertAlmostEqual(float(summary["stfusion_seq_noewc_e100"]["ler_avg_unknown_envs"]), 0.065)
            self.assertAlmostEqual(float(summary["stfusion_seq_noewc_e100"]["ler_worst_env"]), 0.070)
            self.assertLess(
                float(summary["stfusion_seq_noewc_e100"]["delta_vs_stfusion_domestic_e20"]),
                0.0,
            )
            self.assertGreater(len(list(csv.DictReader(detail_csv.open()))), 0)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("seq no-EWC vs domestic 20", text)
            self.assertIn("ST-Fusion-R9-X sequential + EWC epoch 100", text)


if __name__ == "__main__":
    unittest.main()

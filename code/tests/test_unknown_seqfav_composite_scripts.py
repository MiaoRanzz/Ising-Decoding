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
from scripts.generate_unknown_seqfav_composite_configs import (
    DEFAULT_ENV_SPECS,
    generate_seqfav_noise_models,
    write_seqfav_configs,
)
from scripts.run_unknown_seqfav_composite_compare import (
    aggregate_rows,
    load_payloads,
    seqfav_success_status,
    write_csv,
    write_markdown,
)


class TestUnknownSeqfavCompositeConfigGeneration(unittest.TestCase):

    def _base_config_path(self, root: Path) -> Path:
        cfg = {
            "model_id": 111,
            "distance": 9,
            "n_rounds": 9,
            "workflow": {"task": "train"},
            "data": {
                "code_rotation": "O1",
                "noise_model": _single_p_mapping(0.001),
            },
        }
        path = root / "base_t0.yaml"
        OmegaConf.save(OmegaConf.create(cfg), path)
        return path

    def test_composite_envs_use_axis_max_multiplier_without_stacking(self):
        base_noise = _single_p_mapping(0.001)

        generated = generate_seqfav_noise_models(base_noise)

        self.assertEqual([item["env_key"] for item in generated], ["e00", "e01", "e02", "e03", "e04"])
        hard = generated[4]["noise_model"]
        self.assertAlmostEqual(hard["p_meas_X"], base_noise["p_meas_X"] * 1.60)
        self.assertAlmostEqual(hard["p_meas_Z"], base_noise["p_meas_Z"] * 1.35)
        self.assertAlmostEqual(hard["p_cnot_IZ"], base_noise["p_cnot_IZ"] * 1.60)
        self.assertAlmostEqual(hard["p_cnot_IX"], base_noise["p_cnot_IX"] * 1.45)
        self.assertAlmostEqual(hard["p_idle_cnot_Z"], base_noise["p_idle_cnot_Z"] * 1.60)
        self.assertAlmostEqual(hard["p_idle_cnot_X"], base_noise["p_idle_cnot_X"] * 1.35)
        self.assertAlmostEqual(hard["p_prep_X"], base_noise["p_prep_X"] * 1.60)
        self.assertAlmostEqual(hard["p_prep_Z"], base_noise["p_prep_Z"])
        self.assertEqual(generated[4]["axis_multipliers"], DEFAULT_ENV_SPECS[4]["axis_multipliers"])

    def test_written_configs_have_complete_valid_25p_mapping_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_config = self._base_config_path(root)
            paths, manifest = write_seqfav_configs(
                base_config=base_config,
                output_dir=root / "conf" / "experiments" / "unknown_seqfav_composite_v1",
                manifest=root / "manifest.json",
            )

            self.assertEqual([path.name for path in paths], [f"config_unknown_seqfav_composite_v1_e{i:02d}.yaml" for i in range(5)])
            self.assertEqual(manifest["design"], "seq-favoring OOD stress test")
            self.assertEqual(
                manifest["environments"][0]["config_name"],
                "experiments/unknown_seqfav_composite_v1/config_unknown_seqfav_composite_v1_e00",
            )
            self.assertEqual(len(manifest["environments"]), 5)
            self.assertTrue((root / "manifest.json").exists())

            for path in paths:
                cfg = OmegaConf.load(path)
                noise = dict(OmegaConf.to_container(cfg.data.noise_model, resolve=True))
                NoiseModel.from_config_dict(noise)
                self.assertEqual(set(noise), set(_single_p_mapping(0.001)))


class TestUnknownSeqfavCompositeAggregation(unittest.TestCase):

    def _payload(self, config_name: str, noewc_ler: float, ewc_ler: float, domestic100_ler: float) -> dict:
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
                        "logical_errors": int(domestic100_ler * 1000),
                        "samples": 1000,
                        "ler": domestic100_ler,
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
            "config_name": config_name,
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

    def test_markdown_records_seq_favoring_gate_against_domestic100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "payloads"
            output_dir.mkdir()
            envs = ["config_unknown_seqfav_composite_v1_e00", "config_unknown_seqfav_composite_v1_e01"]
            (output_dir / "unknown_seqfav_composite_v1_e00.json").write_text(
                json.dumps(self._payload(envs[0], noewc_ler=0.070, ewc_ler=0.076, domestic100_ler=0.080)),
                encoding="utf-8",
            )
            (output_dir / "unknown_seqfav_composite_v1_e01.json").write_text(
                json.dumps(self._payload(envs[1], noewc_ler=0.060, ewc_ler=0.068, domestic100_ler=0.065)),
                encoding="utf-8",
            )

            payloads = load_payloads(output_dir, envs)
            detail_rows, summary_rows = aggregate_rows(payloads)
            status = seqfav_success_status(summary_rows)
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
            self.assertLess(float(summary["stfusion_seq_noewc_e100"]["delta_vs_stfusion_domestic_e100"]), 0.0)
            self.assertTrue(status["avg_beats_domestic100"])
            self.assertEqual(status["win_count_vs_domestic100"], 2)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("seq-favoring OOD stress test", text)
            self.assertIn("seq no-EWC vs domestic 100", text)
            self.assertIn("domestic-only epoch 100", text)


if __name__ == "__main__":
    unittest.main()

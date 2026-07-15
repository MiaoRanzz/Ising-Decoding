# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_unknown_axismix_grid_u1p2_5p0_ewc_compare import (
    SEQ_EWC_METHOD,
    aggregate_ewc_payloads,
    build_tasks,
)


class TestUnknownAxisMixGridEwcSupplement(unittest.TestCase):
    def _manifest(self):
        envs = []
        for env_index, multiplier_key, multiplier in [(0, "m1p2", 1.2), (10, "m5p0", 5.0)]:
            envs.append(
                {
                    "env_index": env_index,
                    "env_key": f"e{env_index:02d}",
                    "multiplier_key": multiplier_key,
                    "multiplier": multiplier,
                    "config_name": (
                        "experiments/unknown_axismix_grid_u1p2_5p0/"
                        f"config_unknown_axismix_grid_u1p2_5p0_e{env_index:02d}_{multiplier_key}"
                    ),
                    "axis_signature": "meas_all+cnot_all" if env_index == 0 else "meas_all+cnot_all+idle_all+z_bias",
                    "active_axes": ["meas_all", "cnot_all"],
                    "combination_size": 2 if env_index == 0 else 4,
                    "contains_z_bias": env_index == 10,
                    "contains_cnot_z_bias": env_index == 10,
                }
            )
        return {"environments": envs}

    def _payload(self, config_name: str, x: float, z: float) -> dict:
        return {
            "config_name": config_name,
            "rows": [
                {"basis": "X", "method": SEQ_EWC_METHOD, "ler": x, "latency_us_per_round": 2.0, "speedup_vs_pymatching": 1.5},
                {"basis": "Z", "method": SEQ_EWC_METHOD, "ler": z, "latency_us_per_round": 3.0, "speedup_vs_pymatching": 1.6},
            ],
        }

    def test_build_tasks_runs_only_seq_ewc_model_with_distance_specific_outputs(self):
        tasks = build_tasks(
            self._manifest(),
            output_dir=Path("/tmp/ewc_d7"),
            phase="full",
            python="python",
            paired_script=Path("code/scripts/paired_inference_compare.py"),
            gpus=["2", "3"],
            distance=7,
            n_rounds=7,
            num_samples=4096,
            latency_num_samples=512,
            batch_size=2048,
            num_workers=0,
            seed=12345,
            basis="both",
        )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].gpu, "2")
        self.assertIn("--distance", tasks[0].cmd)
        self.assertIn("7", tasks[0].cmd)
        command = " ".join(tasks[0].cmd)
        self.assertIn("stfusion_seq_ewc_e100", command)
        self.assertNotIn("stfusion_seq_noewc_e100", command)
        self.assertTrue(str(tasks[0].output_path).endswith("unknown_axismix_grid_u1p2_5p0_e00_m1p2.json"))

    def test_aggregate_ewc_payloads_computes_x_z_and_both_summary(self):
        manifest = self._manifest()
        payloads = [
            self._payload(Path(env["config_name"]).name, x=0.20, z=0.10)
            for env in manifest["environments"]
        ]

        details, summary = aggregate_ewc_payloads(payloads, manifest, distance=7, run_phase="full")

        self.assertEqual(len(details), 2 * 3)
        first = next(row for row in details if row["basis"] == "both" and row["env_index"] == 0)
        self.assertEqual(first["distance"], 7)
        self.assertAlmostEqual(first["seq_ewc_ler"], 0.15)
        overall = next(row for row in summary if row["group_type"] == "all" and row["basis"] == "both")
        self.assertEqual(overall["config_count"], 2)
        self.assertAlmostEqual(overall["seq_ewc_ler_mean"], 0.15)


if __name__ == "__main__":
    unittest.main()

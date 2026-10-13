# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.experiments.qewc_v3.plot_pilot_tradeoff import points


class TestQewcV3PilotTradeoff(unittest.TestCase):

    def evaluations(self):
        streams = {"T0": {"X": {"sha256": "same"}}}
        ewc = {
            "test_streams": streams,
            "ler_rows": {"C0": [0.1] * 5, "SEQ": [0.2] * 5, "EWC_A": [0.15] * 5},
        }
        replay = {
            "test_streams": streams,
            "ler_rows": {"C0": [0.1] * 5, "SEQ": [0.2] * 5, "R10": [0.17] * 5},
        }
        return ewc, replay

    def test_selected_candidates_are_marked(self):
        ewc, replay = self.evaluations()
        rows = points(ewc, replay, {"selected": {"name": "EWC_A"}}, {"selected": {"name": "R10"}})
        selected = {row["candidate"] for row in rows if row["selected"]}
        self.assertEqual(selected, {"EWC_A", "R10"})

    def test_rejects_different_test_streams(self):
        ewc, replay = self.evaluations()
        replay["test_streams"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "identical"):
            points(ewc, replay, {"selected": {"name": "EWC_A"}}, {"selected": {"name": "R10"}})


if __name__ == "__main__":
    unittest.main()

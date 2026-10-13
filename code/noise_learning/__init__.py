# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noise-parameter learning from hardware syndrome data."""

from .paper_model import NoiseLearningNetwork
from .paper_loss import (
    PAPER_EDGE_TYPES,
    PAPER_HYPEREDGE_TYPES,
    PaperEdgeHyperedgeLoss,
    PaperFormulaCatalog,
    evaluate_xor_formulas,
    xor_probabilities,
)
from .google_qec import (
    GoogleQECDataset,
    GoogleQECExperiment,
    NoiseLearningResult,
    fit_noise_model,
    inject_noise,
)

__all__ = [
    "NoiseLearningNetwork",
    "PaperEdgeHyperedgeLoss",
    "PaperFormulaCatalog",
    "PAPER_EDGE_TYPES",
    "PAPER_HYPEREDGE_TYPES",
    "evaluate_xor_formulas",
    "xor_probabilities",
    "GoogleQECDataset",
    "GoogleQECExperiment",
    "NoiseLearningResult",
    "fit_noise_model",
    "inject_noise",
]

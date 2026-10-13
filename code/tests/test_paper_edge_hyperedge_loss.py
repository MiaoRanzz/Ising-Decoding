# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from noise_learning.paper_loss import (
    PAPER_EDGE_TYPES,
    PAPER_HYPEREDGE_TYPES,
    PARAMETER_NAMES,
    PaperEdgeHyperedgeLoss,
    PaperFormulaCatalog,
    evaluate_xor_formulas,
    xor_probabilities,
)


def _catalog() -> PaperFormulaCatalog:
    # Repeated formulas are intentional: these tests isolate loss algebra and
    # count weighting from a particular device's formula catalog.
    edges = tuple((((i % 25),),) for i in range(PAPER_EDGE_TYPES))
    hypers = tuple(
        (((i % 25, (i + 1) % 25), ((i + 2) % 25,)))
        for i in range(PAPER_HYPEREDGE_TYPES)
    )
    return PaperFormulaCatalog(
        basis="X",
        edge_formulas=edges,
        hyperedge_formulas=hypers,
        edge_counts=tuple(float(i + 1) for i in range(PAPER_EDGE_TYPES)),
        hyperedge_counts=tuple(float(i + 1) for i in range(PAPER_HYPEREDGE_TYPES)),
    )


def test_xor_matches_paper_equation_62() -> None:
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    pair = 0.1 + 0.2 - 2 * 0.1 * 0.2
    expected = pair + 0.3 - 2 * pair * 0.3
    assert float(xor_probabilities(values)) == pytest.approx(expected)


def test_formula_adds_same_location_before_xor() -> None:
    parameters = torch.zeros(25, dtype=torch.float64)
    parameters[0:3] = torch.tensor([0.01, 0.02, 0.03])
    actual = evaluate_xor_formulas(parameters, (((0, 1), (2,)),))[0]
    expected = (0.01 + 0.02) + 0.03 - 2 * (0.01 + 0.02) * 0.03
    assert float(actual) == pytest.approx(expected)


def test_count_weighted_loss_and_gradient() -> None:
    catalog = _catalog()
    loss_fn = PaperEdgeHyperedgeLoss(catalog)
    target = torch.full((25,), 0.003, dtype=torch.float64)
    predicted = torch.full((25,), 0.004, dtype=torch.float64, requires_grad=True)
    breakdown = loss_fn.breakdown(predicted, target)
    predicted_edges, predicted_hypers = catalog.probabilities(predicted)
    target_edges, target_hypers = catalog.probabilities(target)
    edge_counts = torch.tensor(catalog.edge_counts, dtype=torch.float64)
    hyper_counts = torch.tensor(catalog.hyperedge_counts, dtype=torch.float64)
    expected_edge = (edge_counts * (predicted_edges - target_edges).square()).sum()
    expected_hyper = (hyper_counts * (predicted_hypers - target_hypers).square()).sum()
    assert torch.allclose(breakdown.edge, expected_edge)
    assert torch.allclose(breakdown.hyperedge, expected_hyper)
    assert torch.allclose(breakdown.total, expected_edge + expected_hyper)
    breakdown.total.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0


def test_unbiased_loss_uses_equations_66_to_68() -> None:
    catalog = _catalog()
    biased = PaperEdgeHyperedgeLoss(catalog)
    unbiased = PaperEdgeHyperedgeLoss(catalog, unbiased=True, p_min=1e-3, p_max=1e-2)
    target = torch.full((25,), 0.003)
    predicted = torch.full((25,), 0.004)
    base = 0.005
    ratio = (math.sqrt(1e-3 * 1e-2) / base) ** 2
    assert float(unbiased(predicted, target, base_error_rate=base)) == pytest.approx(
        float(biased(predicted, target)) * ratio
    )


def test_catalog_loader_requires_exactly_18_and_43() -> None:
    formula = [[[PARAMETER_NAMES[0]]]]
    payload = {
        "basis": "X",
        "edge_formulas": formula * 17,
        "hyperedge_formulas": formula * 43,
        "edge_counts": [1] * 17,
        "hyperedge_counts": [1] * 43,
    }
    with pytest.raises(ValueError, match="18 edge"):
        PaperFormulaCatalog.from_mapping(payload)

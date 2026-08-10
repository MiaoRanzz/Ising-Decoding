# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable 18-edge/43-hyperedge loss from Chamberland et al.

Every graph probability is an XOR of independent fault locations. Paulis
belonging to the same channel/location are mutually exclusive and are added
first. Independent locations are then XOR-combined. A formula is represented
as ``((parameter_index, ...), ...)``: inner tuples are sums and the outer tuple
is the XOR.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
import yaml

from qec.noise_model import NoiseModel


PAPER_EDGE_TYPES = 18
PAPER_HYPEREDGE_TYPES = 43
PARAMETER_NAMES = tuple(NoiseModel().canonical_parameters())

Component = tuple[int, ...]
XorFormula = tuple[Component, ...]


def xor_probabilities(probabilities: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Odd-parity probability of independent Bernoulli events (Eq. 62)."""

    if probabilities.shape[dim] == 0:
        shape = list(probabilities.shape)
        del shape[dim]
        return torch.zeros(
            shape, dtype=probabilities.dtype, device=probabilities.device
        )
    return 0.5 * (1.0 - torch.prod(1.0 - 2.0 * probabilities, dim=dim))


def evaluate_xor_formulas(
    parameters: torch.Tensor, formulas: Sequence[XorFormula]
) -> torch.Tensor:
    """Evaluate XOR-of-sums formulas on tensors ending in 25 parameters."""

    if parameters.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"expected final parameter dimension {len(PARAMETER_NAMES)}, "
            f"got {parameters.shape[-1]}"
        )
    outputs: list[torch.Tensor] = []
    for formula in formulas:
        components: list[torch.Tensor] = []
        for component in formula:
            if not component:
                raise ValueError("formula components cannot be empty")
            index = torch.as_tensor(
                component, dtype=torch.long, device=parameters.device
            )
            components.append(parameters.index_select(-1, index).sum(dim=-1))
        outputs.append(xor_probabilities(torch.stack(components, dim=-1)))
    if not outputs:
        return parameters.new_empty((*parameters.shape[:-1], 0))
    return torch.stack(outputs, dim=-1)


def _parse_formulas(
    raw: object, *, field: str, parameter_names: Sequence[str]
) -> tuple[XorFormula, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    index = {name: i for i, name in enumerate(parameter_names)}
    parsed: list[XorFormula] = []
    for formula_number, formula in enumerate(raw):
        if not isinstance(formula, Sequence) or isinstance(formula, (str, bytes)):
            raise ValueError(f"{field}[{formula_number}] must be a sequence")
        components: list[Component] = []
        for component_number, component in enumerate(formula):
            if not isinstance(component, Sequence) or isinstance(
                component, (str, bytes)
            ):
                raise ValueError(
                    f"{field}[{formula_number}][{component_number}] must be a sequence"
                )
            try:
                values = tuple(index[str(name)] for name in component)
            except KeyError as exc:
                raise ValueError(f"unknown noise parameter {exc.args[0]!r}") from exc
            if not values:
                raise ValueError("formula components cannot be empty")
            if len(values) != len(set(values)):
                raise ValueError(
                    "a same-location component contains a duplicate parameter"
                )
            components.append(values)
        if not components:
            raise ValueError("formulas cannot be empty")
        parsed.append(tuple(components))
    return tuple(parsed)


@dataclass(frozen=True)
class PaperFormulaCatalog:
    """The 18+43 formulas and graph-instance counts for one logical basis."""

    basis: str
    edge_formulas: tuple[XorFormula, ...]
    hyperedge_formulas: tuple[XorFormula, ...]
    edge_counts: tuple[float, ...]
    hyperedge_counts: tuple[float, ...]
    parameter_names: tuple[str, ...] = PARAMETER_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", self.basis.upper())
        if self.basis not in {"X", "Z"}:
            raise ValueError("basis must be X or Z")
        if tuple(self.parameter_names) != PARAMETER_NAMES:
            raise ValueError(
                "catalog parameter order must match NoiseModel canonical order"
            )
        if len(self.edge_formulas) != PAPER_EDGE_TYPES:
            raise ValueError(f"expected {PAPER_EDGE_TYPES} edge formulas")
        if len(self.hyperedge_formulas) != PAPER_HYPEREDGE_TYPES:
            raise ValueError(f"expected {PAPER_HYPEREDGE_TYPES} hyperedge formulas")
        if len(self.edge_counts) != PAPER_EDGE_TYPES:
            raise ValueError(f"expected {PAPER_EDGE_TYPES} edge counts")
        if len(self.hyperedge_counts) != PAPER_HYPEREDGE_TYPES:
            raise ValueError(f"expected {PAPER_HYPEREDGE_TYPES} hyperedge counts")
        counts = (*self.edge_counts, *self.hyperedge_counts)
        if any(not math.isfinite(v) or v < 0 for v in counts):
            raise ValueError("formula counts must be finite and non-negative")
        if sum(self.edge_counts) <= 0 or sum(self.hyperedge_counts) <= 0:
            raise ValueError(
                "edge and hyperedge counts must each have positive total weight"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PaperFormulaCatalog":
        names = tuple(str(v) for v in payload.get("parameter_names", PARAMETER_NAMES))
        return cls(
            basis=str(payload["basis"]),
            edge_formulas=_parse_formulas(
                payload["edge_formulas"], field="edge_formulas", parameter_names=names
            ),
            hyperedge_formulas=_parse_formulas(
                payload["hyperedge_formulas"],
                field="hyperedge_formulas",
                parameter_names=names,
            ),
            edge_counts=tuple(float(v) for v in payload["edge_counts"]),  # type: ignore[arg-type]
            hyperedge_counts=tuple(float(v) for v in payload["hyperedge_counts"]),  # type: ignore[arg-type]
            parameter_names=names,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PaperFormulaCatalog":
        payload = yaml.safe_load(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("formula catalog root must be a mapping")
        return cls.from_mapping(payload)

    def probabilities(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            evaluate_xor_formulas(parameters, self.edge_formulas),
            evaluate_xor_formulas(parameters, self.hyperedge_formulas),
        )


@dataclass(frozen=True)
class PaperLossBreakdown:
    total: torch.Tensor
    edge: torch.Tensor
    hyperedge: torch.Tensor
    scale: torch.Tensor


class PaperEdgeHyperedgeLoss(nn.Module):
    """Equations (63)-(68), including optional variance stabilisation."""

    def __init__(
        self,
        catalog: PaperFormulaCatalog,
        *,
        unbiased: bool = False,
        p_min: float = 1e-3,
        p_max: float = 1e-2,
    ) -> None:
        super().__init__()
        if not (0 < p_min < p_max):
            raise ValueError("expected 0 < p_min < p_max")
        self.catalog = catalog
        self.unbiased = bool(unbiased)
        self.p0 = math.sqrt(float(p_min) * float(p_max))
        self.register_buffer("edge_counts", torch.tensor(catalog.edge_counts))
        self.register_buffer("hyperedge_counts", torch.tensor(catalog.hyperedge_counts))

    def breakdown(
        self,
        predicted_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
        *,
        base_error_rate: torch.Tensor | float | None = None,
    ) -> PaperLossBreakdown:
        predicted_edges, predicted_hyperedges = self.catalog.probabilities(
            predicted_parameters
        )
        target_edges, target_hyperedges = self.catalog.probabilities(target_parameters)
        edge = (
            (predicted_edges - target_edges).square()
            * self.edge_counts.to(dtype=predicted_edges.dtype)
        ).sum(dim=-1)
        hyperedge = (
            (predicted_hyperedges - target_hyperedges).square()
            * self.hyperedge_counts.to(dtype=predicted_hyperedges.dtype)
        ).sum(dim=-1)
        if self.unbiased:
            if base_error_rate is None:
                raise ValueError("base_error_rate is required for the unbiased loss")
            rate = torch.as_tensor(
                base_error_rate,
                dtype=predicted_parameters.dtype,
                device=predicted_parameters.device,
            )
            if bool(torch.any(rate <= 0)):
                raise ValueError("base_error_rate must be positive")
            scale = (self.p0 / rate).square()
        else:
            scale = predicted_parameters.new_ones(())
        edge = edge * scale
        hyperedge = hyperedge * scale
        return PaperLossBreakdown(
            (edge + hyperedge).mean(), edge.mean(), hyperedge.mean(), scale
        )

    def forward(
        self,
        predicted_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
        *,
        base_error_rate: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        return self.breakdown(
            predicted_parameters, target_parameters, base_error_rate=base_error_rate
        ).total

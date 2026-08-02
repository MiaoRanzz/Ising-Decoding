#!/usr/bin/env python3
"""Generate a fixed, label-preserving synthetic surface-code corpus.

Unlike ``code/export/generate_test_data.py``, this exporter retains the
training tensors.  Row ``i`` in each ``.npy`` file belongs to exactly the same
physical shot:

* ``dets_and_obs.npy``: Stim detectors with appended ground-truth observable;
* ``train_x.npy``: predecoder input, shape ``(N, 4, T, D, D)``;
* ``train_y.npy``: HE-simplified correction label with the same shape.

The dataset is written as NPY memmaps so it can be generated in batches and
later loaded without copying the complete corpus into RAM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from qec.noise_model import NoiseModel
from qec.precompute_dem import precompute_dem_bundle_surface_code
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
from qec.surface_code.stim_sample_io import normalize_code_rotation


def _load_noise_model(config_path: Path | None, simple_noise: bool, p_error: float):
    if simple_noise:
        return None, {"kind": "simple", "p_error": float(p_error)}
    if config_path is None:
        raise ValueError("--config is required unless --simple-noise is used")
    cfg = OmegaConf.load(config_path)
    raw = OmegaConf.select(cfg, "data.noise_model")
    if raw is None:
        raise ValueError(f"{config_path} has no data.noise_model section")
    params = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(params, dict):
        raise ValueError("data.noise_model must be a mapping")
    model = NoiseModel.from_config_dict(params)
    return model, {
        "kind": "noise_model",
        "parameters": model.canonical_parameters(),
        "sha256": model.sha256(),
    }


def _measurements_to_dets_and_obs(
    stim_circuit,
    code,
    meas_old: torch.Tensor,
    x_cum: torch.Tensor,
    z_cum: torch.Tensor,
) -> np.ndarray:
    """Rebuild circuit-order measurements, then use Stim's m2d converter.

    This mirrors the validated conversion in ``test_oracle_predecoder.py``.
    Keeping the conversion here ensures stored labels and stored detector rows
    come from the same Torch-generated shot.
    """
    order: list[tuple[str, int]] = []
    for name, targets, _ in stim_circuit.flattened_operations():
        if name not in ("M", "MX", "MZ", "MR", "MRX", "MRZ"):
            continue
        basis = "X" if name in ("MX", "MRX") else "Z"
        order.extend((basis, int(q)) for q in targets)

    bsz, rounds, _ = meas_old.shape
    data_qubits = np.asarray(code.data_qubits).reshape(-1)
    xchecks = np.asarray(code.xcheck_qubits).reshape(-1)
    zchecks = np.asarray(code.zcheck_qubits).reshape(-1)
    data_index = {int(q): i for i, q in enumerate(data_qubits)}
    xcheck_index = {int(q): i for i, q in enumerate(xchecks)}
    zcheck_index = {int(q): i for i, q in enumerate(zchecks)}

    meas_np = meas_old.detach().cpu().numpy()
    x_np = x_cum.detach().cpu().numpy()
    z_np = z_cum.detach().cpu().numpy()
    measurements = np.zeros((bsz, len(order)), dtype=np.uint8)
    ancilla_per_round = len(xchecks) + len(zchecks)
    for column, (basis, qubit) in enumerate(order):
        if qubit in data_index:
            # An X-basis readout observes Z frame and vice versa.
            frame = z_np if basis == "X" else x_np
            measurements[:, column] = frame[:, -1, data_index[qubit]]
        else:
            round_index = column // ancilla_per_round
            if round_index >= rounds:
                raise ValueError("measurement ordering has more ancilla rounds than generator output")
            local_index = xcheck_index[qubit] if basis == "X" else len(xchecks) + zcheck_index[qubit]
            measurements[:, column] = meas_np[:, round_index, local_index]

    converter = stim_circuit.compile_m2d_converter()
    return np.asarray(
        converter.convert(measurements=np.asarray(measurements, dtype=bool), append_observables=True),
        dtype=np.uint8,
    )


def _make_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)


DEFAULT_SETTINGS = Path(__file__).with_name("end_to_end.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed synthetic shots with paired trainX/trainY.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Workflow settings YAML.")
    parser.add_argument("--output-dir", default=None, help="Override generation.output_dir.")
    parser.add_argument("--project-config", "--config", dest="project_config", type=Path, default=None,
                        help="Override generation.project_config.")
    parser.add_argument("--simple-noise", action="store_true", default=None,
                        help="Use scalar --p-error instead of a YAML noise model.")
    parser.add_argument("--p-error", type=float, default=None)
    parser.add_argument("--distance", type=int, default=None, help="Default: inherit project_config.")
    parser.add_argument("--n-rounds", type=int, default=None, help="Default: inherit project_config.")
    parser.add_argument("--basis", choices=("X", "Z"), default=None)
    parser.add_argument("--code-rotation", default=None, help="Default: inherit project_config data.code_rotation.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu.")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_settings(cli: argparse.Namespace) -> SimpleNamespace:
    settings_path = cli.settings.expanduser().resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"settings file not found: {settings_path}")
    section = OmegaConf.to_container(OmegaConf.load(settings_path).get("generation", {}), resolve=True)
    if not isinstance(section, dict):
        raise ValueError("settings generation section must be a mapping")

    def pick(name: str, *, required: bool = False, fallback=None):
        value = getattr(cli, name)
        if value is None:
            value = section.get(name, fallback)
        if required and value is None:
            raise ValueError(f"missing generation.{name} in {settings_path}")
        return value

    project_config_raw = pick("project_config")
    project_cfg = OmegaConf.load(_repo_path(project_config_raw)) if project_config_raw else None
    distance = pick("distance")
    n_rounds = pick("n_rounds")
    rotation = pick("code_rotation")
    if project_cfg is not None:
        distance = distance if distance is not None else OmegaConf.select(project_cfg, "distance")
        n_rounds = n_rounds if n_rounds is not None else OmegaConf.select(project_cfg, "n_rounds")
        rotation = rotation if rotation is not None else OmegaConf.select(project_cfg, "data.code_rotation")
    if distance is None or n_rounds is None:
        raise ValueError("distance and n_rounds must be set in project_config or settings")
    return SimpleNamespace(
        output_dir=_repo_path(pick("output_dir", required=True)),
        project_config=_repo_path(project_config_raw) if project_config_raw else None,
        simple_noise=bool(pick("simple_noise", fallback=False)),
        p_error=float(pick("p_error", fallback=0.003)),
        distance=int(distance),
        n_rounds=int(n_rounds),
        basis=str(pick("basis", required=True)).upper(),
        code_rotation=str(rotation or "O1"),
        num_samples=int(pick("num_samples", required=True)),
        batch_size=int(pick("batch_size", required=True)),
        seed=int(pick("seed", required=True)),
        device=pick("device"),
    )


def main() -> None:
    args = resolve_settings(parse_args())
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.project_config
    noise_model, noise_metadata = _load_noise_model(config_path, args.simple_noise, args.p_error)
    rotation = normalize_code_rotation(args.code_rotation)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    p_placeholder = float(noise_model.get_max_probability()) if noise_model else float(args.p_error)
    print(f"[prepare] D={args.distance}, T={args.n_rounds}, basis={args.basis}, device={device}")
    artifacts = precompute_dem_bundle_surface_code(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        code_rotation=rotation,
        p_scalar=p_placeholder,
        dem_output_dir=None,
        device=device,
        export=False,
        return_artifacts=True,
        noise_model=noise_model,
    )
    generator = MemoryCircuitTorch(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        code_rotation=rotation,
        H=artifacts["H"],
        p=artifacts["p"],
        A=artifacts.get("A"),
        device=device,
    )
    circuit_wrapper = MemoryCircuit(
        distance=args.distance,
        idle_error=p_placeholder,
        sqgate_error=p_placeholder,
        tqgate_error=p_placeholder,
        spam_error=(2.0 / 3.0) * p_placeholder,
        n_rounds=args.n_rounds,
        basis=args.basis,
        code_rotation=rotation,
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circuit_wrapper.set_error_rates()
    stim_circuit = circuit_wrapper.stim_circuit
    width = int(stim_circuit.num_detectors + stim_circuit.num_observables)
    tensor_shape = (args.num_samples, 4, args.n_rounds, args.distance, args.distance)
    dets_store = _make_memmap(output_dir / "dets_and_obs.npy", (args.num_samples, width))
    input_store = _make_memmap(output_dir / "train_x.npy", tensor_shape)
    label_store = _make_memmap(output_dir / "train_y.npy", tensor_shape)

    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        train_x, train_y, meas_old, x_cum, z_cum = generator.generate_batch(
            batch_size=count, return_aux=True, seed=args.seed + start
        )
        dets_and_obs = _measurements_to_dets_and_obs(
            stim_circuit, generator.code, meas_old, x_cum, z_cum
        )
        if dets_and_obs.shape != (count, width):
            raise RuntimeError(f"unexpected detector shape {dets_and_obs.shape}; expected {(count, width)}")
        end = start + count
        dets_store[start:end] = dets_and_obs
        input_store[start:end] = train_x.detach().to(torch.uint8).cpu().numpy()
        label_store[start:end] = train_y.detach().to(torch.uint8).cpu().numpy()
        if start == 0 or end == args.num_samples or end % (args.batch_size * 32) == 0:
            print(f"[generate] {end}/{args.num_samples} shots")

    del dets_store, input_store, label_store
    metadata = {
        "schema_version": 1,
        "artifact": "labeled_end_to_end_surface_code",
        "num_samples": args.num_samples,
        "distance": args.distance,
        "n_rounds": args.n_rounds,
        "basis": args.basis,
        "code_rotation": rotation,
        "num_detectors": int(stim_circuit.num_detectors),
        "num_observables": int(stim_circuit.num_observables),
        "seed": args.seed,
        "noise": noise_metadata,
        "files": {
            "dets_and_obs": "dets_and_obs.npy",
            "train_x": "train_x.npy",
            "train_y": "train_y.npy",
        },
        "tensor_layout": "(sample, channel, round, row, column)",
        "channels": ["z_data", "x_data", "x_syndrome", "z_syndrome"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote paired corpus to {output_dir}")


if __name__ == "__main__":
    main()

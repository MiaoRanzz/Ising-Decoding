"""Submit the distance-3 surface-code experiment to LQCloud.

The NVIDIA integration imports :func:`run_hardware_experiment`, receives the
returned ``Result``, and consumes ``result.get_memory()`` directly. Importing
this module never submits a job.
"""

from __future__ import annotations

from typing import Iterable, Optional

try:
    from . import circuits, const
except ImportError:  # Support running this file directly.
    import circuits
    import const


def run_hardware_experiment(
    *,
    ini_state: Optional[Iterable[int]] = None,
    cycle: int = 9,
    shots: int = 100,
    backend_name: str = "QZ01-surface_code",
    circuit_type: str = "memory_z",
    provider=None,
):
    """Build and execute the LQCloud circuit, returning the SDK ``Result``.

    ``provider`` is injectable for local tests. When omitted, the same
    ``LQCloudProvider`` construction used by the original script is used.
    """
    if ini_state is None:
        ini_state = [0] * (const.DISTANCE**2)
    ini_state = [int(value) for value in ini_state]
    if len(ini_state) != const.DISTANCE**2 or any(value not in (0, 1) for value in ini_state):
        raise ValueError(
            f"ini_state must contain exactly {const.DISTANCE**2} binary values"
        )
    if int(cycle) < 2:
        raise ValueError("cycle must be at least 2")
    if int(shots) <= 0:
        raise ValueError("shots must be positive")

    qc = circuits.build_cloud_circuit(
        ini_state=ini_state,
        cycle=int(cycle),
        circuit_type=str(circuit_type),
    )
    if provider is None:
        from lqcloud import LQCloudProvider

        provider = LQCloudProvider()
    backend = provider.get_backend(str(backend_name))
    job = backend.run(qc, shots=int(shots))
    return job.result()


def main() -> None:
    cycle = 9
    ini_state = [0] * (const.DISTANCE**2)
    result = run_hardware_experiment(
        ini_state=ini_state,
        cycle=cycle,
        shots=100,
        backend_name="QZ01-surface_code",
        circuit_type="memory_z",
    )

    measurement_bits = result.get_memory()
    print(measurement_bits)

    try:
        from . import data_process
    except ImportError:
        import data_process

    data_process.detection_event_fraction(result, ini_state=ini_state, cycle=cycle)
    logical_error_corrected, logical_error_raw = data_process.logical_error(
        result,
        ini_state=ini_state,
        cycle=cycle,
    )
    print(f"Corrected logical error: {logical_error_corrected:.3f}")
    print(f"Raw logical error: {logical_error_raw:.3f}")


if __name__ == "__main__":
    main()

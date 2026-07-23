"""Generate noisy Stim measurements in the same JSON format as LQCloud output."""

import json
from pathlib import Path

import circuits
import const


# Keep these defaults aligned with get_measurement.py so the generated file can
# be substituted for a hardware measurement file without changing its format.
cycle = 9
shots = 50000
ini_state = [0] * (const.DISTANCE**2)


# build_stim_circuit uses its circuit-level noise defaults.  The compiled
# sampler returns one chronological measurement-record bit per column.
circuit = circuits.build_stim_circuit(
    distance=const.DISTANCE,
    ini_state=ini_state,
    cycle=cycle,
    reset=const.RESET,
    mirror=const.MIRROR,
)
measurements = circuit.compile_sampler().sample(shots=shots)

# LQCloud's get_memory() is serialized as a JSON array of bit strings.  Keep
# the Stim measurement-record order unchanged: it is the order expected by the
# existing data_process.py converter.
data = ["".join("1" if bit else "0" for bit in shot) for shot in measurements]

file_path = Path("lqcloud_simulations/50000x5d5c9/sim_5.json")
file_path.parent.mkdir(parents=True, exist_ok=True)
with file_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

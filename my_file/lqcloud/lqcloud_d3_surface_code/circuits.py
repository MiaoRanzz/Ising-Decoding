"""
qubit index convention:
e.g. d=3 layout:
mirror=True:
-----------------------------------
|           15                    |
|                                 |
|       6        7        8       |
|                                 |
|           11       12      14   |
|         ( Z )    ( X )          |
|       3        4        5       |
|                                 |
|   16       9       10           |
|         (2, 2)                  |
|       0        1        2       |
|    (1, 1)   (1, 3)              |
|                    13           |
-----------------------------------
mirror=False:
-----------------------------------
|                    15           |
|                                 |
|       6        7        8       |
|                                 |
|   16      11       12           |
|                                 |
|       3        4        5       |
|                                 |
|            9       10      14   |
|         (2, 2)                  |
|       0        1        2       |
|    (1, 1)   (1, 3)              |
|           13                    |
-----------------------------------

stim coord convention:
mirror=True:
-----------------------------------
| -->x      15                    |
| |                               |
| v     6        7        8       |
| y                               |
|           11       12      14   |
|                                 |
|       3        4        5       |
|                                 |
|   16       9       10           |
|         (2, -2)                 |
|       0        1        2       |
|    (1, -1)   (3, -1)            |
|                    13           |
-----------------------------------


Chip coord convention:
mirror=True:
-----------------------------------
|           15                    |
|                                 |
|       6        7        8       |
|                                 |
|           11       12      14   |
|                                 |
|       3        4        5       |
|                                 |
|   16       9       10           |
|         (2, 0)                  |
|       0        1        2       |
|    (0, 0)   (2, -2)             |
|                    13           |
-----------------------------------
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

try:
    from . import const
except ImportError:  # Support running this directory's scripts directly.
    import const
import numpy as np
import stim

try:
    from lqcloud import QuantumCircuit
except ImportError:  # Stim-only decoding does not require the cloud SDK.
    QuantumCircuit = None


def stim_coord_convention(coord: Tuple[int, int]) -> Tuple[int, int]:
    return (coord[1], -coord[0])


def generate_qubit_coords(
    distance: int, mirror: bool = True
) -> Dict[int, Tuple[int, int]]:
    coords = {}
    # data qubits
    for i in range(distance):
        for j in range(distance):
            coords[i * distance + j] = (2 * i + 1, 2 * j + 1)
    # center measure qubits
    for i in range(distance - 1):
        for j in range(distance - 1):
            coords[i * (distance - 1) + j + distance**2] = (2 * i + 2, 2 * j + 2)

    boundary_measure_qubit_num = round((distance - 1) / 2)
    boundary_measure_qubit_start_index = distance**2 + (distance - 1) ** 2
    # bottom
    for i in range(boundary_measure_qubit_num):
        if mirror:
            coords[i + boundary_measure_qubit_start_index] = (0, 4 * i + 4)
        else:
            coords[i + boundary_measure_qubit_start_index] = (0, 4 * i + 2)
    # right
    for i in range(boundary_measure_qubit_num):
        if mirror:
            coords[
                i + boundary_measure_qubit_num + boundary_measure_qubit_start_index
            ] = (4 * i + 4, 2 * distance)
        else:
            coords[
                i + boundary_measure_qubit_num + boundary_measure_qubit_start_index
            ] = (4 * i + 2, 2 * distance)
    # top
    for i in range(boundary_measure_qubit_num):
        if mirror:
            coords[
                i + boundary_measure_qubit_num * 2 + boundary_measure_qubit_start_index
            ] = (2 * distance, 2 * distance - 4 * i - 4)
        else:
            coords[
                i + boundary_measure_qubit_num * 2 + boundary_measure_qubit_start_index
            ] = (2 * distance, 2 * distance - 4 * i - 2)
    # left
    for i in range(boundary_measure_qubit_num):
        if mirror:
            coords[
                i + boundary_measure_qubit_num * 3 + boundary_measure_qubit_start_index
            ] = (2 * distance - 4 * i - 4, 0)
        else:
            coords[
                i + boundary_measure_qubit_num * 3 + boundary_measure_qubit_start_index
            ] = (2 * distance - 4 * i - 2, 0)
    return coords


def generate_cz_pattern_and_stabilizer_qubits(
    qubit_coords: Dict[int, Tuple[int, int]],
    z_measure_qubits: Iterable[int],
    x_measure_qubits: Iterable[int],
    mirror: bool,
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    coord2qubit = {coord: qubit for qubit, coord in qubit_coords.items()}
    cz_pattern1 = []
    cz_pattern2 = []
    cz_pattern3 = []
    cz_pattern4 = []
    stabilizer_qubits = {}

    def handle_measure_qubit(measure_qubit, measure_qubit_type: str):
        _stabilizer_qubits = []
        coord = qubit_coords[measure_qubit]
        coord_lu = (coord[0] + 1, coord[1] - 1)
        coord_ru = (coord[0] + 1, coord[1] + 1)
        coord_ld = (coord[0] - 1, coord[1] - 1)
        coord_rd = (coord[0] - 1, coord[1] + 1)
        if coord_lu in coord2qubit:
            cz_pattern1.append((measure_qubit, coord2qubit[coord_lu]))
            _stabilizer_qubits.append(coord2qubit[coord_lu])
        if coord_ru in coord2qubit:
            if measure_qubit_type == "z":
                cz_pattern2.append((measure_qubit, coord2qubit[coord_ru]))
            elif measure_qubit_type == "x":
                cz_pattern3.append((measure_qubit, coord2qubit[coord_ru]))
            else:
                raise KeyError(measure_qubit_type)
            _stabilizer_qubits.append(coord2qubit[coord_ru])
        if coord_ld in coord2qubit:
            if measure_qubit_type == "z":
                cz_pattern3.append((measure_qubit, coord2qubit[coord_ld]))
            elif measure_qubit_type == "x":
                cz_pattern2.append((measure_qubit, coord2qubit[coord_ld]))
            else:
                raise KeyError(measure_qubit_type)
            _stabilizer_qubits.append(coord2qubit[coord_ld])
        if coord_rd in coord2qubit:
            cz_pattern4.append((measure_qubit, coord2qubit[coord_rd]))
            _stabilizer_qubits.append(coord2qubit[coord_rd])
        stabilizer_qubits[measure_qubit] = _stabilizer_qubits

    for qubit in z_measure_qubits:
        handle_measure_qubit(qubit, "z")
    for qubit in x_measure_qubits:
        handle_measure_qubit(qubit, "x")

    if mirror:
        cz_patterns = [cz_pattern1, cz_pattern3, cz_pattern2, cz_pattern4]
    else:
        cz_patterns = [cz_pattern1, cz_pattern2, cz_pattern3, cz_pattern4]
    return cz_patterns, stabilizer_qubits


def generate_z_x_measure_qubits(distance: int, mirror: bool):
    boundary_measure_qubit_num = round((distance - 1) / 2)
    z_measure_qubits = []
    x_measure_qubits = []
    for _row in range(distance - 1):
        _measure_qubits = np.arange(distance - 1) + (distance - 1) * _row + distance**2
        if _row % 2 == 0:
            z_measure_qubits.append(_measure_qubits[1::2])
            x_measure_qubits.append(_measure_qubits[::2])
        else:
            z_measure_qubits.append(_measure_qubits[::2])
            x_measure_qubits.append(_measure_qubits[1::2])
    horizontal_boundary_measure_qubits = np.hstack(
        [
            np.arange(boundary_measure_qubit_num) + distance**2 + (distance - 1) ** 2,
            np.arange(boundary_measure_qubit_num)
            + boundary_measure_qubit_num * 2
            + distance**2
            + (distance - 1) ** 2,
        ]
    )
    vertical_boundary_measure_qubits = np.hstack(
        [
            np.arange(boundary_measure_qubit_num)
            + boundary_measure_qubit_num
            + distance**2
            + (distance - 1) ** 2,
            np.arange(boundary_measure_qubit_num)
            + boundary_measure_qubit_num * 3
            + distance**2
            + (distance - 1) ** 2,
        ]
    )
    if mirror:
        z_measure_qubits.append(vertical_boundary_measure_qubits)
        x_measure_qubits.append(horizontal_boundary_measure_qubits)
    else:
        z_measure_qubits.append(horizontal_boundary_measure_qubits)
        x_measure_qubits.append(vertical_boundary_measure_qubits)
    z_measure_qubits = np.hstack(z_measure_qubits)
    x_measure_qubits = np.hstack(x_measure_qubits)

    return z_measure_qubits, x_measure_qubits


def build_stim_circuit(
    distance: int,
    ini_state: Iterable[int],
    cycle: int = 3,
    circuit_type="memory_z",
    reset=True,
    sq_error: Union[float, Dict[int, float]] = 0.0011,
    cz_error: Union[float, Dict[int, float]] = 0.0064,
    measure_error: Union[float, Dict[int, float]] = 0.015,
    idle_z_error: Union[float, Dict[int, float]] = 0.0,
    idle_dep_error: Union[float, Dict[int, float]] = 0.019,
    mirror=True,
) -> stim.Circuit:
    assert cycle >= 2
    assert circuit_type == "memory_z" or circuit_type == "memory_x"
    data_qubits = np.arange(distance**2)
    assert len(ini_state) == len(data_qubits)
    measure_qubits = np.arange(distance**2 - 1) + distance**2

    data_qubits, measure_qubits = list(data_qubits), list(measure_qubits)
    all_qubits = data_qubits + measure_qubits

    measure_qubit_num = len(measure_qubits)
    data_qubit_num = len(data_qubits)

    qubit_coords = generate_qubit_coords(distance=distance, mirror=mirror)

    z_measure_qubits, x_measure_qubits = generate_z_x_measure_qubits(distance, mirror)
    cz_patterns, stabilizer_qubits = generate_cz_pattern_and_stabilizer_qubits(
        qubit_coords=qubit_coords,
        z_measure_qubits=z_measure_qubits,
        x_measure_qubits=x_measure_qubits,
        mirror=mirror,
    )
    cz_patterns = [np.hstack(pattern) for pattern in cz_patterns]

    if circuit_type == "memory_z":
        qubits_apply_H = data_qubits[1::2]
        detector_qubits = z_measure_qubits
    else:
        qubits_apply_H = data_qubits[::2]
        detector_qubits = x_measure_qubits

    circuit = stim.Circuit()
    for qubit, coord in qubit_coords.items():
        circuit.append("QUBIT_COORDS", qubit, stim_coord_convention(coord))
    # prepare initial state
    targets_1 = []
    for data_qubit, s in zip(data_qubits, ini_state):
        if s == 1:
            targets_1.append(data_qubit)
    if len(targets_1) > 0:
        circuit.append("TICK")
        circuit.append("X", targets_1)
        circuit.append("DEPOLARIZE1", targets_1, sq_error)

    circuit.append("TICK")
    circuit.append("H", qubits_apply_H)
    circuit.append("DEPOLARIZE1", qubits_apply_H, sq_error)

    # cycle
    for cycle_idx in range(cycle):
        circuit.append("TICK")
        circuit.append("H", measure_qubits)
        # circuit.append("X", data_qubits)
        # circuit.append("DEPOLARIZE1", all_qubits, sq_error)
        circuit.append("DEPOLARIZE1", measure_qubits, sq_error)

        circuit.append("TICK")
        circuit.append("CZ", cz_patterns[0])
        circuit.append("DEPOLARIZE2", cz_patterns[0], cz_error)

        circuit.append("TICK")
        # circuit.append("X", measure_qubits)
        circuit.append("H", data_qubits)
        # circuit.append("DEPOLARIZE1", all_qubits, sq_error)
        circuit.append("DEPOLARIZE1", data_qubits, sq_error)

        circuit.append("TICK")
        circuit.append("CZ", cz_patterns[1])
        circuit.append("DEPOLARIZE2", cz_patterns[1], cz_error)

        circuit.append("TICK")
        circuit.append("X", all_qubits)
        circuit.append("DEPOLARIZE1", all_qubits, sq_error)

        circuit.append("TICK")
        circuit.append("CZ", cz_patterns[2])
        circuit.append("DEPOLARIZE2", cz_patterns[2], cz_error)

        circuit.append("TICK")
        # circuit.append("X", measure_qubits)
        circuit.append("H", data_qubits)
        # circuit.append("DEPOLARIZE1", all_qubits, sq_error)
        circuit.append("DEPOLARIZE1", data_qubits, sq_error)

        circuit.append("TICK")
        circuit.append("CZ", cz_patterns[3])
        circuit.append("DEPOLARIZE2", cz_patterns[3], cz_error)

        if cycle_idx == cycle - 1:  # last cycle
            circuit.append("TICK")
            circuit.append("H", measure_qubits)
            circuit.append("H", qubits_apply_H)
            circuit.append("DEPOLARIZE1", measure_qubits, sq_error)
            circuit.append("DEPOLARIZE1", qubits_apply_H, sq_error)

            circuit.append("TICK")
            circuit.append("X_ERROR", all_qubits, measure_error)
            circuit.append("M", measure_qubits)
        else:
            circuit.append("TICK")
            circuit.append("H", measure_qubits)
            # circuit.append("X", data_qubits)
            # circuit.append("DEPOLARIZE1", all_qubits, sq_error)
            circuit.append("DEPOLARIZE1", measure_qubits, sq_error)

            circuit.append("TICK")
            circuit.append("X_ERROR", measure_qubits, measure_error)
            if reset:
                circuit.append("MR", measure_qubits)
            else:
                circuit.append("M", measure_qubits)
            circuit.append("Z_ERROR", data_qubits, idle_z_error)
            circuit.append("DEPOLARIZE1", data_qubits, idle_dep_error)

        if reset:
            if cycle_idx == 0:  # first cycle
                for qubit in detector_qubits:
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(
                                -measure_qubit_num + measure_qubits.index(qubit)
                            )
                        ],
                    )
            else:  # subsequent rounds
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-measure_qubit_num + idx),
                            stim.target_rec(-2 * measure_qubit_num + idx),
                        ],
                    )
        else:
            if cycle_idx == 0:  # first cycle
                for qubit in detector_qubits:
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(
                                -measure_qubit_num + measure_qubits.index(qubit)
                            )
                        ],
                    )
            elif cycle_idx == 1:  # second cycle
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-measure_qubit_num + idx),
                        ],
                    )
            else:  # subsequent rounds
                for idx in range(len(measure_qubits)):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-measure_qubit_num + idx),
                            stim.target_rec(-3 * measure_qubit_num + idx),
                        ],
                    )

    circuit.append("M", data_qubits)
    if reset:
        for qubit in detector_qubits:
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(
                        -data_qubit_num + data_qubits.index(_stabilizer_qubit)
                    )
                    for _stabilizer_qubit in stabilizer_qubits[qubit]
                ]
                + [
                    stim.target_rec(
                        -data_qubit_num
                        - measure_qubit_num
                        + measure_qubits.index(qubit)
                    )
                ],
            )
    else:
        for qubit in detector_qubits:
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(
                        -data_qubit_num + data_qubits.index(_stabilizer_qubit)
                    )
                    for _stabilizer_qubit in stabilizer_qubits[qubit]
                ]
                + [
                    stim.target_rec(
                        -data_qubit_num
                        - measure_qubit_num
                        + measure_qubits.index(qubit)
                    ),
                    stim.target_rec(
                        -data_qubit_num
                        - 2 * measure_qubit_num
                        + measure_qubits.index(qubit)
                    ),
                ],
            )
    if (circuit_type == "memory_z" and mirror) or (
        circuit_type == "memory_x" and not mirror
    ):
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-data_qubit_num + idx) for idx in range(distance)],
            0,
        )
    else:
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [
                stim.target_rec(-data_qubit_num + idx * distance)
                for idx in range(distance)
            ],
            0,
        )

    return circuit


def build_cloud_circuit(
    ini_state: Iterable[int],
    cycle: int = 3,
    circuit_type: str = "memory_z",
    distance: int | None = None,
) -> QuantumCircuit:
    if QuantumCircuit is None:
        raise ImportError(
            "build_cloud_circuit requires the lqcloud SDK. "
            "Install lqcloud to submit circuits; build_stim_circuit works without it."
        )
    if distance is None:
        distance = const.DISTANCE
    distance = int(distance)
    reset = const.RESET
    mirror = const.MIRROR
    assert cycle >= 2
    assert circuit_type == "memory_z" or circuit_type == "memory_x"

    data_qubits = np.arange(distance**2)
    measure_qubits = np.arange(distance**2 - 1) + distance**2
    data_qubits, measure_qubits = (
        [int(qubit) for qubit in data_qubits],
        [int(qubit) for qubit in measure_qubits],
    )
    data_qubit_num = len(data_qubits)
    measure_qubit_num = len(measure_qubits)
    all_qubits = measure_qubits + data_qubits

    qubit_coords = generate_qubit_coords(distance=distance, mirror=mirror)

    z_measure_qubits, x_measure_qubits = generate_z_x_measure_qubits(distance, mirror)
    z_measure_qubits = [int(qubit) for qubit in z_measure_qubits]
    x_measure_qubits = [int(qubit) for qubit in x_measure_qubits]
    cz_patterns, _ = generate_cz_pattern_and_stabilizer_qubits(
        qubit_coords=qubit_coords,
        z_measure_qubits=z_measure_qubits,
        x_measure_qubits=x_measure_qubits,
        mirror=mirror,
    )

    if circuit_type == "memory_z":
        qubits_apply_H = data_qubits[1::2]
    else:
        qubits_apply_H = data_qubits[::2]

    circuit = QuantumCircuit(
        data_qubit_num + measure_qubit_num, data_qubit_num + measure_qubit_num * cycle
    )
    # prepare initial state
    for data_qubit, s in zip(data_qubits, ini_state):
        if s == 1:
            circuit.x(data_qubit)

    circuit.barrier(all_qubits)
    circuit.h(qubits_apply_H)

    # cycle
    for cycle_idx in range(cycle):
        circuit.barrier(all_qubits)
        circuit.h(measure_qubits)

        circuit.barrier(all_qubits)
        for measure_qubit, data_qubit in cz_patterns[0]:
            circuit.cz(measure_qubit, data_qubit)

        circuit.barrier(all_qubits)
        circuit.h(data_qubits)

        circuit.barrier(all_qubits)
        for measure_qubit, data_qubit in cz_patterns[1]:
            circuit.cz(measure_qubit, data_qubit)

        circuit.barrier(all_qubits)
        circuit.x(all_qubits)

        circuit.barrier(all_qubits)
        for measure_qubit, data_qubit in cz_patterns[2]:
            circuit.cz(measure_qubit, data_qubit)

        circuit.barrier(all_qubits)
        circuit.h(data_qubits)

        circuit.barrier(all_qubits)
        for measure_qubit, data_qubit in cz_patterns[3]:
            circuit.cz(measure_qubit, data_qubit)

        if cycle_idx == cycle - 1:  # last cycle
            circuit.barrier(all_qubits)
            circuit.h(measure_qubits + qubits_apply_H)

            circuit.barrier(all_qubits)
            circuit.measure(
                measure_qubits + data_qubits,
                [
                    int(cbit)
                    for cbit in np.arange(measure_qubit_num + data_qubit_num)
                    + measure_qubit_num * cycle_idx
                ],
            )
        else:
            circuit.barrier(all_qubits)
            for qubit in measure_qubits:
                circuit.h(qubit)

            circuit.barrier(all_qubits)
            circuit.timeanchor(measure_qubits[0], "DD_start")
            circuit.measure(
                measure_qubits,
                [
                    int(cbit)
                    for cbit in np.arange(measure_qubit_num)
                    + measure_qubit_num * cycle_idx
                ],
            )
            if reset:
                raise Exception("Reset will be supported in the future.")
                circuit.reset(measure_qubits)
            circuit.barrier(measure_qubits)
            circuit.timeanchor(measure_qubits[0], "DD_end")
            for qubit in data_qubits:
                circuit.dd(qubit, "DD_start", "DD_end", m=8)
    return circuit

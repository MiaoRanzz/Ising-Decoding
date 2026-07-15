from typing import Dict, Iterable

import circuits
import const
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatching
from numpy.typing import NDArray
from scipy.optimize import least_squares

from lqcloud.job.result import Result


def convert_to_01(result: Result) -> NDArray:
    measurements = []
    for bs, count in result.get_counts().items():
        measurements.append([[(int(b)) for b in bs]] * count)
    return np.vstack(measurements)


def detection_event_fraction(
    result: Result, ini_state: Iterable[int], cycle: int, circuit_type: str = "memory_z"
):
    # Get parameters
    distance = const.DISTANCE
    reset = const.RESET
    mirror = const.MIRROR
    data_qubits = np.arange(distance**2)
    measure_qubits = np.arange(distance**2 - 1) + distance**2
    data_qubits, measure_qubits = (
        [int(qubit) for qubit in data_qubits],
        [int(qubit) for qubit in measure_qubits],
    )

    # Get detector_qubits
    z_measure_qubits, x_measure_qubits = circuits.generate_z_x_measure_qubits(
        distance=distance, mirror=mirror
    )
    if circuit_type == "memory_z":
        detector_qubits = z_measure_qubits
    else:
        detector_qubits = x_measure_qubits
    detector_qubits = [measure_qubits[qubit - distance**2] for qubit in detector_qubits]
    # Get weight_2_measure_qubits and weight_4_measure_qubits
    _, stabilizer_qubits = circuits.generate_cz_pattern_and_stabilizer_qubits(
        qubit_coords=circuits.generate_qubit_coords(distance=distance, mirror=mirror),
        z_measure_qubits=z_measure_qubits,
        x_measure_qubits=x_measure_qubits,
        mirror=mirror,
    )
    weight_2_measure_qubits = []
    weight_4_measure_qubits = []
    for _measure_qubit, _data_qubits in stabilizer_qubits.items():
        if len(_data_qubits) == 2:
            weight_2_measure_qubits.append(measure_qubits[_measure_qubit - distance**2])
        elif len(_data_qubits) == 4:
            weight_4_measure_qubits.append(measure_qubits[_measure_qubit - distance**2])
        else:
            raise RuntimeError(_data_qubits)

    circuit = circuits.build_stim_circuit(
        distance,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        mirror=mirror,
    )
    measurements = convert_to_01(result)

    converter = circuit.compile_m2d_converter()
    detection_events, _ = converter.convert(
        measurements=measurements.astype(np.bool_), separate_observables=True
    )
    def_mean = np.mean(detection_events, axis=0)

    def reshape_def(def_array):
        def_dict = {
            0: dict(zip(detector_qubits, def_array[: len(detector_qubits)])),
            cycle: dict(zip(detector_qubits, def_array[-len(detector_qubits) :])),
        }
        for _cycle in range(1, cycle):
            def_dict[_cycle] = dict(
                zip(
                    measure_qubits,
                    def_array[
                        len(detector_qubits) + len(measure_qubits) * (_cycle - 1) : len(
                            detector_qubits
                        )
                        + len(measure_qubits) * _cycle
                    ],
                )
            )
        def_df = pd.DataFrame(def_dict).T.sort_index()
        return def_df

    reshaped_def_mean = reshape_def(def_mean)

    colors = {"weight-4": "#1a73e8", "weight-2": "#12b5cb"}

    plt.figure(figsize=[4, 3])
    for _qubit in measure_qubits:
        if _qubit in weight_2_measure_qubits:
            c = colors["weight-2"]
        elif _qubit in weight_4_measure_qubits:
            c = colors["weight-4"]
        else:
            raise RuntimeError(_qubit)
        plt.plot(
            reshaped_def_mean[_qubit].index,
            reshaped_def_mean[_qubit],
            marker="o",
            markersize=1,
            color=c,
            alpha=0.1,
        )
    plt.errorbar(
        reshaped_def_mean[weight_2_measure_qubits].index,
        reshaped_def_mean[weight_2_measure_qubits].mean(axis=1),
        reshaped_def_mean[weight_2_measure_qubits].std(axis=1),
        marker="o",
        markersize=1,
        capsize=0,
        color=colors["weight-2"],
        label="weight-2",
    )
    plt.errorbar(
        reshaped_def_mean[weight_4_measure_qubits].index,
        reshaped_def_mean[weight_4_measure_qubits].mean(axis=1),
        reshaped_def_mean[weight_4_measure_qubits].std(axis=1),
        marker="o",
        markersize=1,
        capsize=0,
        color=colors["weight-4"],
        label="weight-4",
    )

    plt.xlabel("Cycle")
    plt.ylabel("Detection event fraction")
    plt.legend()
    plt.ylim(0, 0.5)
    plt.title("ini state: " + "".join([str(s) for s in ini_state]))
    plt.tight_layout()
    plt.show()


def fit_cycle_raise(x, y):
    def fit_func(p, x):
        return 0.5 - 0.5 * (1 - 2 * p[0]) ** x

    def error(p):
        return y - fit_func(p, x)

    bounds = [[0], [0.5]]
    p0 = [max(y) / max(x)]

    result_opt = least_squares(error, p0, bounds=bounds)

    def fit_func_opt(x):
        return fit_func(result_opt.x, x)

    return result_opt, fit_func_opt


def _measurements_to_logical_error(
    measurements: NDArray,
    cycle: int,
    distance: int,
    ini_state: Iterable[int],
    circuit_type: str,
    reset: bool,
    mirror: bool,
    **kwargs,
):
    """
    :params measurements: shape:[shots, record_idx]
    """
    circuit = circuits.build_stim_circuit(
        distance=distance,
        ini_state=ini_state,
        cycle=cycle,
        circuit_type=circuit_type,
        reset=reset,
        mirror=mirror,
        **kwargs,
    )
    converter = circuit.compile_m2d_converter()
    detection_events, observable_flips = converter.convert(
        measurements=measurements.astype(np.bool_), separate_observables=True
    )
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model()
    )
    predictions = matcher.decode_batch(detection_events)

    logical_error_corrected = 1 - np.mean(observable_flips == predictions)
    logical_error = np.mean(observable_flips)
    return logical_error_corrected, logical_error


def logical_error(
    result: Result,
    ini_state: Iterable[int],
    cycle: int,
    circuit_type: str = "memory_z",
    **kwargs,
):
    distance = const.DISTANCE
    reset = const.RESET
    mirror = const.MIRROR

    logical_error_corrected, logical_error = _measurements_to_logical_error(
        measurements=convert_to_01(result),
        cycle=cycle,
        distance=distance,
        ini_state=ini_state,
        circuit_type=circuit_type,
        reset=reset,
        mirror=mirror,
        **kwargs,
    )
    return logical_error_corrected, logical_error

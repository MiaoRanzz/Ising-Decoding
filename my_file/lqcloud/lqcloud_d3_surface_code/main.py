import os
import sys

# LQCloud provider for submitting jobs to the quantum backend
from lqcloud import LQCloudProvider
# Local helpers: circuit generation and result decoding/plotting
import circuits
import data_process
import const

# Number of stabilizer measurement rounds
# cycle = 10
cycle = 9
# Initial state of the 9 data qubits: all |0>
# ini_state = [0] * 9
ini_state = [0] * (const.DISTANCE ** 2)

# Build the distance-3 surface code cloud circuit
qc = circuits.build_cloud_circuit(ini_state=ini_state, cycle=cycle)

# Connect to LQCloud and select the target backend
provider = LQCloudProvider()
backend = provider.get_backend("QZ01-surface_code")
# backend = provider.get_backend("AGate-100")

# Submit the job and wait for the result (1024 shots)
job = backend.run(qc, shots=100)
result = job.result()

# Each entry is a full measurement record for one shot, e.g. a 81-bit string for d=3 surface code.
measurement_bits = result.get_memory()
print(measurement_bits)

# Plot the detection event fraction as a function of stabilizer round
data_process.detection_event_fraction(result, ini_state=ini_state, cycle=cycle)

# Decode the result and compute corrected / raw logical error rates
logical_error_corrected, logical_error_raw = data_process.logical_error(
    result, ini_state=ini_state, cycle=cycle
)

# print("###############################################################")
# print("###############################################################")
# print("###############################################################")
# print(result.data)
# print("###############################################################")
# print("###############################################################")
# print("###############################################################")
# print(result.get_memory())
# print("###############################################################")
# print("###############################################################")
# print("###############################################################")
# print(result.print_result())
# print("###############################################################")
# print("###############################################################")
# print("###############################################################")

# Print the two logical error rates with three decimal places
print(f"Corrected logical error: {logical_error_corrected:.3f}")
print(f"Raw logical error: {logical_error_raw:.3f}")

###############################################################
###############################################################
###############################################################
# Integration into NVIDIA's inference pipeline

# from code.workflows.run import run

# run()
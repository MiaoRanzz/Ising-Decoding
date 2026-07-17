from lqcloud import LQCloudProvider
import circuits
import const
import json


# Number of stabilizer measurement rounds
cycle = 9
ini_state = [0] * (const.DISTANCE ** 2)

# Build the distance-3 surface code cloud circuit
qc = circuits.build_cloud_circuit(ini_state=ini_state, cycle=cycle)

# Connect to LQCloud and select the target backend
provider = LQCloudProvider()
backend = provider.get_backend("QZ01-surface_code")
# backend = provider.get_backend("AGate-100")

# Submit the job and wait for the result
job = backend.run(qc, shots=10000)
result = job.result()

data = result.get_memory()
file_path = "lqcloud_measurements/measurement_5.json"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
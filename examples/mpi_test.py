from NoisyCircuits import QuantumCircuitMPI as QC
from NoisyCircuits.utils.CreateNoiseModel import GetNoiseModel, CreateNoiseModel
import pickle
import os
import numpy as np
import json
import time


backend_name = "ibm_fez"
num_qubits = 20
num_cores = 2
num_ranks = 25
threshold = 1e-6
verbose = True
# Options: "heron", "eagle". Note that "eagle" is now deprecated and only available in simulation mode (i.e., no noise model from hardware).
qpu_type = "heron" 
sim_backend = "custom" # Choose between "custom", "qulacs", "pennylane" and "qiskit"
use_fractional = True # Boolean Flag to determine whether to use fractional gates or not.

if qpu_type == "eagle":
    file_path = "https://raw.githubusercontent.com/Sats2/NoisyCircuits/main/noise_models/Noise_Model_Eagle_QPU.pkl"
    noise_model = pickle.load(open(file_path, "rb"))
elif qpu_type == "heron":
    file_path = "https://raw.githubusercontent.com/Sats2/NoisyCircuits/main/noise_models/Sample_Noise_Model_Heron_QPU.csv"
    noise_model = CreateNoiseModel(calibration_data_file=file_path, 
                                   basis_gates=[["x", "sx", "rz", "rx"], ["cz", "rzz"]]).create_noise_model()
else:
    raise ValueError("Invalid qpu_type. Choose either 'heron' or 'eagle'.")


def build_circuit(circuit):
    num_qubits = circuit.num_qubits
    single_qubit_gates = ["x", "sx", "rx", "rz"]
    for _ in range(100):
        for q in range(num_qubits):
            choice = np.random.choice(single_qubit_gates)
            if choice == "x":
                circuit.X(q)
            elif choice == "sx":
                circuit.SX(q)
            elif choice == "rz":
                circuit.RZ(np.random.uniform(-2*np.pi, 2*np.pi), q)
            else:
                circuit.RX(np.random.uniform(-2*np.pi, 2*np.pi), q)
        for q in range(num_qubits - 1):
            circuit.CZ(q, q + 1)
    return circuit

circ = QC(
    num_qubits = num_qubits,
    noise_model = noise_model,
    use_fractional = use_fractional,
    backend_qpu_type = qpu_type,
    cores_per_rank = num_cores,
    threshold = threshold,
    verbose = verbose
)

circ = build_circuit(circ)
t0 = time.perf_counter_ns()
probs = circ.execute(list(range(num_qubits)), 1000)
t1 = time.perf_counter_ns()
if probs is not None:
    print("Completed! --> Probs (for |00..0> and |11..1> only): {} and {}".format(probs[0], probs[-1]))
    print("Runtime: {}".format((t1 - t0)*1e-9))
    assert np.isclose(probs.sum(), 1.0), "Sum of Probabilities is not 1"
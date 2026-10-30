# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module provides the ability for the QuantumCircuit module to perform a density matrix simulation for a specific quantum circuit using qiskit as a quantum circuit simulation backend. Alternatively, the user can opt to use just this method to perform a density matrix simulation using a custom instruction set with a custom noise model for single qubit and two qubit gates, as long as the gates applied belong to the set of gates pre-defined by the QPU basis gates from the IBM Eagle/Heron QPU architectures. The measurement error is not considered in this module and is only applied from within the QuantumCircuit module. For the full application of noise information from the quantum hardware, it is recommended to run all simulations via the QuantumCircuit module. 

Example:
    >>> import numpy as np
    >>> from NoisyCircuits.utils.qiskit import DensityMatrixSolver
    >>> instruction_list = []
    >>> instruction_list.append(["rx", [0], np.pi])
    >>> instruction_list.append(["ecr", [0, 1], None])
    >>> solver = DensityMatrixSolver(num_qubits=2, single_qubit_noise=single_qubit_noise, two_qubit_noise=two_qubit_noise, instruction_list=instruction_list, num_cores=1)
    >>> solver.solve(qubits=[0,1])
    [0.45, 0.45, 0.05, 0.05]

This module contains only one class `DensityMatrixSolver` which has only one callable function `solve(qubits)` whose arguements are the qubits that are to be measured.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, kraus_error
from qiskit_aer.library import SaveDensityMatrix
from qiskit.quantum_info import partial_trace
import numpy as np


class DensityMatrixSolver:
    """
    Class to solve quantum circuits using density matrices. Assumes that the circuit is defined with the qubit map already implemented.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 instruction_list:list,
                 num_cores:int)->None:
        """
        Initializes the DensityMatrixSolver with the given parameters.

        Parameters:
        -----------
        num_qubits : int
            Number of qubits in the circuit.
        single_qubit_noise : dict
            Noise instructions for single qubit gates for all qubits used.
        two_qubit_noise : dict
            Noise instructions for entangling gates for all qubits used.
        instruction_list : list
            List of instructions to be executed on the circuit.
        """
        self.num_qubits = num_qubits
        self.instruction_list = instruction_list
        self.num_cores = num_cores
        self.noise = NoiseModel()
        for qubit in single_qubit_noise:
            for gate in single_qubit_noise[qubit]:
                error = kraus_error(single_qubit_noise[qubit][gate])
                self.noise.add_quantum_error(error, gate, [qubit])
        for gate in two_qubit_noise:
            for qubit_pair in two_qubit_noise[gate]:
                error = kraus_error(two_qubit_noise[gate][qubit_pair])
                self.noise.add_quantum_error(error, gate, list(qubit_pair))
    
    def solve(self,
              qubits:list[int])->np.ndarray[np.float64]:
        """
        Solves the quantum circuit using density matrix simulation and returns the probabilities of measuring the specified qubits in the computational basis.

        Parameters
        ----------
        qubits : list[int]
            List of qubits to be measured.
        
        Returns
        -------
        np.ndarray[np.float64]
            Probabilities of measuring each qubit in the computational basis.
        """
        circuit = QuantumCircuit(self.num_qubits)
        trace_qubits = [i for i in range(self.num_qubits) if i not in qubits]
        instruction_map = {
            "x": lambda q, p: circuit.x(q[0]),
            "sx": lambda q,p: circuit.sx(q[0]),
            "rz": lambda q, p: circuit.rz(p, q[0]),
            "rx": lambda q, p: circuit.rx(p, q[0]),
            "ecr": lambda q, p: circuit.ecr(q[0], q[1]),
            "cz": lambda q, p: circuit.cz(q[0], q[1]),
            "rzz": lambda q, p: circuit.rzz(p, q[0], q[1]),
            "unitary": lambda q, p: circuit.unitary(p, q)
        }
        for entry in self.instruction_list:
            gate_name = entry[0]
            qubit_index = entry[1]
            parameter = entry[2]
            instruction_map[gate_name](qubit_index, parameter)
        sim = AerSimulator(noise_model=self.noise)
        sim.set_options(max_parallel_threads=self.num_cores)
        circuit.append(SaveDensityMatrix(self.num_qubits), circuit.qubits)
        res = sim.run(circuit).result()
        if len(trace_qubits) == 0:
            probs = np.diag(np.asarray(res.data()["density_matrix"])).real
            return np.require(probs, dtype=np.float64, requirements=["C"])
        probs = np.diag(np.asarray(partial_trace(res.data()["density_matrix"], trace_qubits), order="C")).real
        return np.require(probs, dtype=np.float64, requirements=["C"])
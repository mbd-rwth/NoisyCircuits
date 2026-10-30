# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using qulacs as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from qulacs import QuantumState, QuantumCircuit
import qulacs.gate as gate
import numpy as np
import ray
from NoisyCircuits.utils import compute_trajectory_probs_single, compute_trajectory_probs_two_q, update_state_inplace_1q, update_state_inplace_2q


@ray.remote
class RemoteExecutor:
    """
    Module that performs the parallel execution of the quantum circuit according to the Monte-Carlo Wavefunction method.
    """
    def __init__(self,
                 num_qubits:int,
                 single_qubit_noise:dict,
                 two_qubit_noise:dict,
                 )->None:
        """
        Constructor for the RemoteExecutor class.

        Parameters
        ----------
        num_qubits : int
            Number of qubits in the quantum circuit.
        single_qubit_noise : dict
            Dictionary containing the noise operators for single qubit gates.
        two_qubit_noise : dict
            Dictionary containing the noise operators for two qubit gates.
        """
        self.num_qubits = num_qubits
        self.single_qubit_noise = single_qubit_noise
        self.two_qubit_noise = two_qubit_noise
        self.two_qubit_noise_index = {}
        for k in range(len(self.two_qubit_noise)):
            self.two_qubit_noise_index[self.two_qubit_noise[k][0]] = k
        self.probs_sum = np.zeros(2**self.num_qubits, dtype=np.float64)
        exp = lambda x: np.exp(1j * x)
        self.instruction_map = {
            "x": lambda q, p: gate.X(q[0]),
            "sx": lambda q, p: gate.sqrtX(q[0]),
            "rz": lambda q, p: gate.RotZ(q[0], p),
            "rx": lambda q, p: gate.RotX(q[0], p),
            "cz": lambda q, p: gate.CZ(q[0], q[1]),
            "ecr": lambda q, p: gate.DenseMatrix(q, (1 / np.sqrt(2)) * np.array([[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]])),
            "rzz": lambda q, p: gate.DenseMatrix(q, np.array([[exp(-p/2), 0, 0, 0], [0, exp(p/2), 0, 0], [0, 0, exp(p/2), 0], [0, 0, 0, exp(-p/2)]])),
            "unitary": lambda q, p: gate.DenseMatrix(q, p) if len(q) > 1 else gate.DenseMatrix(q[0], p)
        }
        self.noise_function_map = {
            "x": self._apply_single_qubit_noise,
            "sx": self._apply_single_qubit_noise,
            "rz": self._apply_single_qubit_noise,
            "rx": self._apply_single_qubit_noise,
            "cz": self._apply_two_qubit_noise,
            "ecr": self._apply_two_qubit_noise,
            "rzz": self._apply_two_qubit_noise,
            "unitary": self._no_noise
        }
    
    def _apply_single_qubit_noise(self,
                                   state:np.ndarray[np.complex128],
                                   gate_name:str,
                                   qubit_index:list[int]
                                   )->np.ndarray[np.complex128]:
        """
        Private method that applies applies the noise operator for single qubit gates.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise must be applied to.
        
        Returns
        -------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        ops = self.single_qubit_noise[qubit_index[0]][1][gate_name]
        kraus_probs = compute_trajectory_probs_single(ops, state, qubit_index[0])
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        update_state_inplace_1q(ops[chosen_idx], state, qubit_index[0])
        return state / np.sqrt(kraus_probs[chosen_idx])
    
    def _apply_two_qubit_noise(self,
                               state:np.ndarray[np.complex128],
                               gate_name:str,
                               qubit_index:list[int]
                               )->np.ndarray[np.complex128]:
        """
        Private method that applies the noise operator for two qubit gates.

        Parameters
        ----------
        state : np.ndarray[np.complex128])
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise must be applied to.
        
        Returns
        -------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        qubit_pair = tuple(qubit_index)
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate_name]][1][qubit_pair]
        kraus_probs = compute_trajectory_probs_two_q(ops, state, qubit_index)
        chosen_idx = np.random.choice(len(kraus_probs), p=kraus_probs)
        update_state_inplace_2q(ops[chosen_idx], state, qubit_index[0], qubit_index[1])
        return state / np.sqrt(kraus_probs[chosen_idx])
    
    def _no_noise(self,
                  state:np.ndarray[np.complex128],
                  gate_name:str,
                  qubit_index:list[int]
                )->np.ndarray[np.complex128]:
        """
        Private method that returns the statevector unchanged when no noise is to be applied.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate previously applied.
        qubit_index : list[int]
            List of qubits that the noise would be applied to if there was noise.

        Returns
        -------
        np.ndarray[np.complex128]
            Unchanged statevector since no noise is applied.
        """
        return state

    def run(self,
            traj_id:int,
            instruction_list:list[list[str, list[int], float|None]]
        )->None:
        """
        Main method of the module to execute the MCWF trajectories.

        Parameters
        ----------
        traj_id : int
            Trajectory ID for the simulation.
        instruction_list : list[list[str, list[int], float|None]]
            List of instructions to build the quantum circuit.

        Returns
        -------
        None
        """
        self.instruction_list = instruction_list

        def compute_trajectory(
                traj_id:int
                )->np.ndarray[np.float64]:
            """
            Method to compute a single MCWF trajectory.

            Parameters
            ----------
            traj_id : int
                Trajectory ID for the simulation.

            Returns
            -------
            np.ndarray[np.float64]
                Probabilities after executing the trajectory.
            """
            np.random.seed(42 + traj_id)
            init_state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            init_state[0] = 1.0 + 0.0j
            for gate_name, qubit_index, parameters in self.instruction_list:
                state = QuantumState(self.num_qubits)
                state.load(init_state)
                circuit = QuantumCircuit(self.num_qubits)
                circuit.add_gate(self.instruction_map[gate_name](qubit_index, parameters))
                circuit.update_quantum_state(state)
                state_vector = state.get_vector()
                state_vector_updated = self.noise_function_map[gate_name](state_vector.copy(), gate_name, qubit_index)
                init_state = state_vector_updated
            return init_state

        final_state = compute_trajectory(traj_id)
        self.probs_sum += np.abs(final_state)**2

    def get(self)->np.ndarray[np.float64]:
        """
        Method to get the accumulated probabilities after all trajectories have been run.

        Returns
        -------
        np.ndarray[np.float64]
            Accumulated probabilities after all trajectories.
        """
        return self.probs_sum
    
    def reset(self)->None:
        """
        Method to reset the accumulated probabilities and the measured qubits.
        """
        self.probs_sum = np.zeros(2**self.num_qubits, dtype=np.float64)
        
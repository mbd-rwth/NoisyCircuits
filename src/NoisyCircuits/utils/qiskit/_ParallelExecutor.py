# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using pennylane as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
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
                 two_qubit_noise:dict
                )->None:
        """
        Constructor for the Remote Executor class.

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
        self.circuit = QuantumCircuit(num_qubits)
        self.noise_function_map = {
            "x": self._apply_single_qubit_noise,
            "sx": self._apply_single_qubit_noise,
            "rz": self._apply_single_qubit_noise,
            "rx": self._apply_single_qubit_noise,
            "cz": self._apply_two_qubit_noise,
            "ecr": self._apply_two_qubit_noise,
            "rzz": self._apply_two_qubit_noise,
            "unitary": self._apply_no_noise
        }
        self.instruction_map = {
            "x": lambda q, p: self.circuit.x(q[0]),
            "sx": lambda q, p: self.circuit.sx(q[0]),
            "rz": lambda q, p: self.circuit.rz(p, q[0]),
            "rx": lambda q, p: self.circuit.rx(p, q[0]),
            "ecr": lambda q, p: self.circuit.ecr(q[0], q[1]),
            "cz": lambda q, p: self.circuit.cz(q[0], q[1]),
            "rzz": lambda q, p: self.circuit.rzz(p, q[0], q[1]),
            "unitary": lambda q, p: self.circuit.unitary(p, q)
        }
        self.sim = AerSimulator(method='statevector')
        self.sim.set_options = {"max_parallel_threads" : 1}
        self.trace_qubits = []
    
    def _apply_no_noise(self,
                        state:np.ndarray[np.complex128],
                        gate_name:str,
                        qubit_index:list[int])->np.ndarray[np.complex128]:
        """
        Private method that returns the statevector unchanged when no noise is to be applied.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system.
        gate_name : str
            The name of the gate being applied.
        qubit_index : list[int]
            The list of qubits that the gate is being applied to.

        Returns
        -------
        np.ndarray[np.complex128]
            The unchanged statevector when no noise is applied.
        """
        return state
    
    def _apply_single_qubit_noise(self,
                                  state:np.ndarray[np.complex128],
                                  gate_name:str,
                                  qubit_index:list[int])->np.ndarray[np.complex128]:
        """
        Private method that applies single qubit noise to the given statevector according to the MCWF method.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate being applied.
        qubit_index : list[int]
            List containing the index of the qubit being acted upon
        
        Returns
        --------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        ops = self.single_qubit_noise[qubit_index[0]][1][gate_name]
        kraus_probs = compute_trajectory_probs_single(ops, state, qubit_index[0])
        chosen_index = np.random.choice(len(ops), p=kraus_probs)
        update_state_inplace_1q(ops[chosen_index], state, qubit_index[0])
        return state / np.sqrt(kraus_probs[chosen_index])
    
    def _apply_two_qubit_noise(self,
                               state:np.ndarray[np.complex128],
                               gate_name:str,
                               qubit_index:list[int])->np.ndarray[np.complex128]:
        """
        Private method that applies two qubit noise to the given statevector according to the MCWF method.

        Parameters
        ----------
        state : np.ndarray[np.complex128])
            Current statevector of the quantum system.
        gate_name : str
            Name of the gate being applied.
        qubit_index : list[int]
            List containing the index of the qubit being acted upon
        
        Returns
        --------
        np.ndarray[np.complex128]
            Updated statevector after applying the noise operator.
        """
        qubit_pair = tuple(qubit_index)
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate_name]][1][qubit_pair]
        kraus_probs = compute_trajectory_probs_two_q(ops, state, qubit_index)
        chosen_index = np.random.choice(len(ops), p=kraus_probs)
        update_state_inplace_2q(ops[chosen_index], state, qubit_index[0], qubit_index[1])
        return state / np.sqrt(kraus_probs[chosen_index])
    
    def run(self,
            traj_id:int,
            instruction_list:list[list[str, list[int], float]|None]
            )->None:
        """
        Main method of the module to execute the MCWF trajectories.

        Parameters
        ----------
        traj_id : int
            Trajectory ID for the simulation.
        instruction_list : list[list[str, list[int], float|None]]
            List of instructions to build the quantum circuit.
        """
        self.instruction_list = instruction_list

        def compute_trajectory(traj_id:int)->np.ndarray[np.float64]:
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
            system_qubit_list = list(range(self.num_qubits))
            init_state[0] = 1.0 + 0.0j
            current_state = init_state.copy()
            self.circuit.initialize(init_state, system_qubit_list, normalize=False)
            for entry in self.instruction_list:
                new_circuit = self.circuit.copy_empty_like()
                self.circuit = new_circuit
                gate_name, qubit_index, parameter = entry
                self.circuit.initialize(current_state, system_qubit_list, normalize=False)
                self.instruction_map[gate_name](qubit_index, parameter)
                self.circuit.save_statevector()
                job = self.sim.run(self.circuit).result()
                result = np.asarray(job.get_statevector(), dtype=np.complex128)
                new_state = self.noise_function_map[gate_name](result, gate_name, qubit_index)
                current_state = new_state.copy()
            return current_state
        
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
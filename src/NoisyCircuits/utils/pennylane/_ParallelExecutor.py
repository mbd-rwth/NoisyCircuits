# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module is responsible for running the MCWF simulations in parallel for noisy quantum circuit simulations using pennylane as a quantum circuit simulator backend. It is not meant to be called independently by a user but instead to be used as a helper module within the QuantumCircuit module to perform MCWF trajectory simulations in a parallel (shared/distributed memory) environment.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pennylane as qml
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
        self.measured_qubits = num_qubits
        self.probs_sum = np.zeros(2**self.measured_qubits, dtype=np.float64)
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits)
        self.instruction_map = {
            "x" : lambda p, q: qml.X(wires=q[0]),
            "sx" : lambda p, q: qml.SX(wires=q[0]),
            "rx": lambda p, q: qml.RX(p, wires=q[0]),
            "rz" : lambda p, q: qml.RZ(p, wires=q[0]),
            "ecr" : lambda p, q: qml.ECR(wires=q),
            "cz" : lambda p, q: qml.CZ(wires=q),
            "rzz" : lambda p, q: qml.IsingZZ(p, wires=q),
            "unitary" : lambda p, q: qml.QubitUnitary(p, wires=q)
        }
        self.noise_map = {
            "x" : self._apply_single_qubit_noise,
            "sx" : self._apply_single_qubit_noise,
            "rx" : self._apply_single_qubit_noise,
            "rz" : self._apply_single_qubit_noise,
            "ecr" : self._apply_two_qubit_noise,
            "cz" : self._apply_two_qubit_noise,
            "rzz" : self._apply_two_qubit_noise,
            "unitary" : self._apply_no_noise
        }

    def _apply_single_qubit_noise(self,
                                  state : np.ndarray[np.complex128],
                                  gate : str,
                                  qubit : list[int]
                                )->np.ndarray[np.complex128]:
        """
        Private method to apply the single qubit noise to the statevector.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system after applying the gate.
        gate : str
            The gate for which the noise is to be applied.
        qubit : list[int]
            The qubit to which the noise is applied.

        Returns
        -------
        np.ndarray[np.complex128]
            The updated state of the qubit system after applying the noise operator.
        """
        ops = self.single_qubit_noise[qubit[0]][1][gate]
        bit = self.num_qubits - 1 - qubit[0]
        kraus_probs = compute_trajectory_probs_single(ops, state, bit)
        chosen_index = np.random.choice(len(ops), p = kraus_probs)
        update_state_inplace_1q(ops[chosen_index], state, bit)
        return state / np.sqrt(kraus_probs[chosen_index])
    
    def _apply_two_qubit_noise(self,
                               state:np.ndarray[np.complex128],
                               gate:str,
                               qubit_index:list[int]
                            )->np.ndarray[np.complex128]:
        """
        Private method to apply the two qubit noise to the statevector.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system after applying the gate.
        gate : str
            The gate for which the noise is to be applied.
        qubit_index : list[int]
            The two qubits to which the noise is applied.

        Returns
        -------
        np.ndarray[np.complex128]
            The updated state of the qubit system after applying the two qubit noise operator.
        """
        qubit_pair = tuple(qubit_index)
        ops = self.two_qubit_noise[self.two_qubit_noise_index[gate]][1][qubit_pair]
        bits = [
            self.num_qubits - 1 - qubit_index[1],
            self.num_qubits - 1 - qubit_index[0]
        ]
        kraus_probs = compute_trajectory_probs_two_q(ops, state, bits)
        chosen_index = np.random.choice(len(ops), p = kraus_probs)
        update_state_inplace_2q(ops[chosen_index], state, bits[0], bits[1])
        return state / np.sqrt(kraus_probs[chosen_index])
    
    def _apply_no_noise(self,
                        state : np.ndarray[np.complex128],
                        gate : str,
                        qubit : list[int]
                    )->np.ndarray[np.complex128]:
        """
        Private method to apply no noise to the statevector.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system after applying the gate.
        gate : str
            The gate for which no noise is to be applied.
        qubit : list[int]
            The qubits to which the gate is applied.

        Returns
        -------
        np.ndarray[np.complex128]
            The state unchanged.

        Notes
        -----
        This method is used for gates for which no noise is to be applied, such as the unitary gate in order to preserve the workflow of applying noise after each gate.
        """
        return state
    
    def _apply_gate(self,
                    state : np.ndarray[np.complex128],
                    gate : str,
                    qubit_index : list[int],
                    param : float | np.ndarray[np.complex128] | None
                )->np.ndarray[np.complex128]:
        """
        Private method to apply the gate to the quantum circuit.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system.
        gate : str
            The name of the gate to be applied.
        qubit_index : list[int]
            The qubits to which the gate is applied.
        param : float | np.ndarray[np.complex128] | None
            The parameter for the gate, if applicable.

        Returns
        -------
        np.ndarray[np.complex128]
            The updated state of the qubit system after applying the gate.
        """
        @qml.qnode(self.dev)
        def apply_gate(
            state:np.ndarray[np.complex128],
            gate:str,
            qubit_index:list[int],
            param:float | np.ndarray[np.complex128] | None
        )->np.ndarray[np.complex128]:
            """
            Function that applies the gate to the quantum circuit using pennylane.

            Parameters
            ----------
            state : np.ndarray[np.complex128]
                The current state of the qubit system.
            gate : str
                The name of the gate to be applied.
            qubit_index : list[int]
                The qubits to which the gate is applied.
            param : float | np.ndarray[np.complex128] | None
                The parameter for the gate, if applicable.
            
            Returns
            -------
            np.ndarray[np.complex128]
                The updated state of the qubit system after applying the gate.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            self.instruction_map[gate](param, qubit_index)
            return qml.state()
        return apply_gate(state, gate, qubit_index, param)
    
    def _get_probs(self,
                   state:np.ndarray[np.complex128]
                )->np.ndarray[np.float64]:
        """
        Private method to get the probabilities of the statevector.

        Parameters
        ----------
        state : np.ndarray[np.complex128]
            The current state of the qubit system.

        Returns
        -------
        np.ndarray[np.float64]
            The probabilities of the statevector.
        """
        @qml.qnode(self.dev)
        def get_probs(state:np.ndarray[np.complex128])->np.ndarray[np.float64]:
            """
            Function to get the probabilities of the statevector using pennylane.

            Parameters
            ----------
            state : np.ndarray[np.complex128]
                The current state of the qubit system.
            
            Returns
            -------
            np.ndarray[np.float64]
                The probabilities of the statevector.
            """
            qml.StatePrep(state, wires=range(self.num_qubits))
            return qml.probs(wires=range(self.num_qubits))
        return get_probs(state)

    def run(self,
            traj_id: int,
            instruction_list: list[tuple[str, list[int], float]]
            )->None:
        """
        Main method of the module to execute the MCWF trajectories

        Parameters
        ----------
        traj_id : int
            The trajectory id for which the simulation is run.
        instruction_list : list[tuple[str, list[int], float]]
            List of instructions to build the quantum circuit.
        """
        self.instruction_list = instruction_list

        def compute_trajectory(traj_id:int)->np.ndarray[np.float64]:
            """
            Method to compute a single trajectory of the simulation.

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
            state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            state[0] = 1.0
            for instruction in self.instruction_list:
                gate, qubits, param = instruction
                state = self._apply_gate(state.copy(), gate, qubits, param)
                state = self.noise_map[gate](state, gate, qubits)
            probs = self._get_probs(state)
            return probs
        self.probs_sum += compute_trajectory(traj_id)     

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
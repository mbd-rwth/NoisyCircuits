# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module allows users to create and simulate quantum circuits with noise models based on quantum machine calibration data in a distributed memory environment using MPI. It provides methods for adding gates, executing the circuit with Monte-Carlo simulations, and visualizing the circuit. It considers both single and two-qubit gate errors as well as measurement errors. The functionality is similar to the `QuantumCircuit` module.\n

Example:\n
"""

import warnings
import numpy as np
import simulator_mpi as simulator
import measurement_error_applicator
from collections.abc import Callable
from NoisyCircuits.utils import BuildModel, Parser, compute_marginal_probs, convert_matrix_to_little_endian, basis_gate_set


class QuantumCircuitMPI:
    r"""
    This class allows a user to create a quantum circuit with error models imported either via API or via tabulated data for any gate-based quantum hardware that uses qubits. The in-built functionalities for gate decompositions and hardware mapping is currently restricted to IBM devices (as data is relatively easy to access from public access points) but users can use self composed circuits with hardware selections via importing the quantum circuit either via OpenQASM 3.0 data format or from qiskit quantum circuits.

    Only a limited amount of quantum circuits are available at the moment but supported gates are constantly added and updated.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the quantum circuit.
    noise_model : dict
        A dictionary containing the raw noise model for the quantum circuit.
    use_fractional : bool (Optional)
        A flag to determine whether to use fractional gates or not. Applicable to QPUs which allow fractional gates (Defaults to True)
    backend_qpu_type : str (Optional)
        The QPU architecture to use. Default options are IBM's Eagle and Heron QPUs. Additional QPUs can be temporarily added along with imported circuits. See tutorial about this. (Defaults to heron)
    cores_per_rank : int (Optional)
        Number of CPU cores to use in each rank (Defaults to 1)
    threshold : float (Optional)
        The threshold for pruning errors in the noise model (Defaults to 1e-12)
    verbose : bool (Optional)
        A flag to determine whether to print verbose outputs during the noise model construction. (Defaults to True)

    Raises
    ------
    TypeError : 
        - num_qubits must be an integer
        - noise_model must be a dictionary
        - use_fractional must be a bool
        - backend_qpu_type must be a string
        - threshold must be a float
        - verbose must be a boolean.
    ValueError :
        - num_qubits must be a positive integer
        - backend_qpu_type must be a defined QPU available within the package or ammendable to the package
        - cores_per_rank must be a positive integer
        - threshold must be between 0 and 1    

    Warnings
    --------
    RuntimeWarning
        When a decomposition for the set QPU does not exist and must rely on an imported quantum circuit.    
    """
    def __init__(
                self,
                num_qubits : int,
                noise_model : dict,
                use_fractional : bool = True,
                backend_qpu_type : str = "heron",
                cores_per_rank : int = 1,
                threshold : float = 1e-12,
                verbose : bool = False
    ) -> None:
        """
        Initializes the QuantumCircuitMPI class with the specified number of qubits, noise model and other parameters.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be of type int")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer")
        if not isinstance(noise_model, dict):
            raise TypeError("noise_model must be of type dict")
        if not isinstance(use_fractional, bool):
            raise TypeError("use_fractional must be of type bool")
        if not isinstance(backend_qpu_type, str):
            raise TypeError("backend_qpu_type must be of type string")
        if backend_qpu_type.lower() not in list(basis_gate_set.keys()):
            raise ValueError("backend_qpu_type must be in {}".format(list(basis_gate_set.keys())))
        if not isinstance(cores_per_rank, int):
            raise TypeError("cores_per_rank must be of type int")
        if cores_per_rank <= 0:
            raise ValueError("cores_per_rank must be a positive integer")
        if not isinstance(threshold, float):
            raise TypeError("threshold must be of type float")
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be of type bool")
        self.num_qubits = num_qubits
        self.use_fractional = use_fractional
        self.qpu = backend_qpu_type
        self.cores_per_rank = cores_per_rank
        self.threshold = threshold
        self.verbose = verbose
        self._basis_gates = basis_gate_set[self.qpu]["basis_gates"]
        self.basis_gates = self._basis_gates
        modeller = BuildModel(
                noise_model = noise_model,
                num_qubits = self.num_qubits,
                num_cores = self.cores_per_rank,
                threshold = self.threshold,
                basis_gates = self.basis_gates,
                verbose = self.verbose
        )
        single_error, double_error, measurement_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_error = {
            q : {gate : payload["qubit_channel"] for gate, payload in gates.items()} for q, gates in single_error.items()
        }
        self.two_qubit_error = {
            gate : {pair : convert_matrix_to_little_endian(payload["qubit_channel"]) for pair, payload in pairs.items()} for gate, pairs in double_error.items()
        }
        self.measurement_error = measurement_error
        self.connectivity = connectivity
        if basis_gate_set[self.qpu]["gate_decomposition"] is not None:
            self._gate_decomposition = basis_gate_set[self.qpu]["gate_decomposition"](
                num_qubits = self.num_qubits,
                connectivity = self.connectivity,
                qubit_map = modeller.qubit_coupling_map,
                use_fractional = self.use_fractional
            )
        else:
            self._gate_decomposition = None
            warnings.warn("A decomposition for the given QPU does not exist and therefore, circuit building is not possible. Please import your circuit either via a OpenQasm file or as a Qiskit object.", RuntimeWarning)

    def __getattr__(
            self,
            name :str
    ) -> Callable:
        """
        Delegate unknown attributes / methods to the selected methods class

        Parameters
        ----------
        name : str
            Class name key
        
        Returns
        -------
        Callable
            The method corresponding to the QPU name if it exists in the gate decomposition.

        Raises
        ------
        AttributeError
            If the QPU name does not contain a gate decomposition. 
        """
        if name is not None:
            return getattr(self._gate_decomposition, name)

    @property
    def basis_gates(self) -> list[list[str]]:
        """
        Getter for the basis gates attribute

        Returns
        -------
        list[list[str]]
            The list of basis gates supported by the quantum hardware being simulator.
        """
        return self._basis_gates

    def refresh(self) -> None:
        """
        Clear the quantum circuit instructions
        """
        try:
            self._gate_decomposition.instruction_list = []
        except:
            try:
                self.instruction_list = []
            except:
                print("No instruction list found for given decomposition. Please import a circuit.")

    def _check_gates_in_noise_model(self) -> None:
        """
        Helper method to check if the gates used in the imported quantum circuit are supported by the noise model.

        Raises
        ------
        ValueError
            If any gates used in the quantum circuit are not supported by the noise model.
        """
        single_qubit_gates_in_model = self.single_qubit_error[0].keys()
        two_qubit_gates_in_model = self.two_qubit_error.keys()
        gate_list = [gate for sublist in [single_qubit_gates_in_model, two_qubit_gates_in_model] for gate in sublist]
        unsupported_gates = []
        for instruction in self.instruction_list:
            if instruction[0] not in gate_list and instruction[0] != "unitary":
                unsupported_gates.append(instruction[0])
        if len(unsupported_gates) != 0:
            raise ValueError("The following gates are unsupported by the noise model: {}\nSupported gate list: {}".format(set(unsupported_gates), gate_list))

    def read_openqasm(
            self,
            file_path : str,
            append_to_circuit : bool = False
    ) -> None:
        """
        Reads an OpenQASM 3.0 file and added the required instructions to build the quantum circuit.

        Parameters
        ----------
        file_path : str
            The path to the OpenQASM 3.0 file.
        append_to_circuit : bool (Optional)
            A flag to determine whether add the imported circuit to the existing circuit or replace the existing circuit. (Defaults to True)
        """
        try:
            parser = Parser(
                file_path = file_path,
                instruction_list = self._gate_decomposition.instruction_list,
                append_to_circuit = append_to_circuit,
                basis_gates = self.basis_gates
            )
            self._gate_decomposition.instruction_list = parser.parse()
        except:
            self.instruction_list = []
            parser = Parser(
                file_path = file_path,
                instruction_list = self.instruction_list,
                append_to_circuit = append_to_circuit,
                basis_gates = self.basis_gates
            )
            self.instruction_list = parser.parse()
        self._check_gates_in_noise_model()

    def read_from_qiskit():
        raise NotImplementedError()

    def execute(
            self,
            qubits : list[int] = None,
            num_trajectories : int = 100
    ) -> np.ndarray[np.float64]:
        """
        Simulate the quantum circuit using the MCWF method in an MPI environment with each rank running a single trajectory utilizing multiple cores within the rank. The trajectory distribution within the ranks is performed automatically.

        Parameters
        ----------
        qubits : list[int] (Optional)
            A list of qubit indices that are to measured (Default to None, in which case all qubits are measured)
        num_trajectories : int (Optional)
            The number of Monte-Carlo trajectories to run. (Defaults to 100)
        
        Returns
        -------
        np.ndarray[np.float64]
            The final probabilities of the executed quantum circuit.

        Raises
        ------
        TypeError :
            - qubits must be a list of integers
            - num_trajectories must be an integer.
        ValueError
            - All measurable qubits must be in the range 1 to num_qubits.
            - num_trajectories must be a positive integer.

        Warnings
        --------
        UserWarning
            num_trajectories is too low
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("qubits must be a list of integers")
        if any((q < 1 or q > self.num_qubits) for q in qubits):
            raise ValueError(f"One or more of the qubits are out of range. The valid range is 0 to ·{self.num_qubits}")
        if not isinstance(num_trajectories, int):
            raise TypeError("num_trajectories must be of type int")
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be a positive integer.")
        if num_trajectories < 50:
            warnings.warn("The specified number of trajectories might not be sufficient to produce high-fidelity with a density matrix simulation.")
        simulator.initialize_mpi()
        rank = simulator.get_mpi_rank()
        output_array = simulator.simulate_circuit(
            self.instruction_list,
            self.single_qubit_error,
            self.two_qubit_error,
            self.num_qubits,
            num_trajectories,
            self.cores_per_rank
        )
        simulator.finalize_mpi()
        if rank == 0:
            if len(qubits) < self.num_qubits:
                output_array = compute_marginal_probs(
                    output_array,
                    [q for q in range(self.num_qubits) if q not in qubits]
                )
            measurement_error_applicator.apply_measurement_error(
                output_array,
                self.measurement_error,
                qubits,
                len(qubits),
                self.cores_per_rank
            )
            output_array = output_array.reshape([2] * len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
            return output_array
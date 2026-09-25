# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module allows users to create and simulate quantum circuits with noise models based on quantum machine calibration data. It provides methods for adding gates, executing the circuit with Monte-Carlo simulations, and visualizing the circuit. It considers both single and two-qubit gate errors as well as measurement errors.\n

Example:\n
    >>> from NoisyCircuits.QuantumCircuit import QuantumCircuit
    >>> circuit = QuantumCircuit(num_qubits=3, noise_model=my_noise_model, backend_qpu_type='Heron', use_fractional=True, sim_backend="custom", threshold=1e-8, verbose=False) # Using the default value for basis_gates --> basis gates of the Heron QPU.
    >>> circuit.h(0)
    >>> circuit.cx(0, 1)
    >>> circuit.cx(1, 2)
    >>> circuit.run_with_density_matrix(qubits=[0, 1, 2]) # Executes the circuit using the density matrix solver
    [0.39841323, 0.00300163, 0.09303931, 0.00615167, 0.00616272, 0.09281154, 0.00300024, 0.39741967]
    >>> circuit.execute(qubits=[0, 1, 2]) # Executes the circuit using the Monte-Carlo Wavefunction method
    [0.39748485, 0.0037614 , 0.09168292, 0.00799886, 0.00746056, 0.09156762, 0.00367236, 0.39637143]
    >>> circuit.run_pure_state(qubits=[0, 1, 2], num_cores=2, return_statevector=False) # Executes the circuit using the pure state solver to return probabilities
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    >>> circuit.run_pure_state(qubits=[0, 1, 2], num_cores=2, return_statevector=False) # Executes the circuit using the pure state solver to return amplitudes
    [0.7071067811865475 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0.7071067811865475 + 0j]
"""

import numpy as np
from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
from NoisyCircuits.utils.EagleDecomposition import EagleDecomposition
from NoisyCircuits.utils.HeronDecomposition import HeronDecomposition
from NoisyCircuits.utils import Parser
from NoisyCircuits.utils.solvers import load_solver
from NoisyCircuits.utils import compute_marginal_probs, convert_matrix_to_little_endian, basis_gate_set
import measurement_error_applicator
import ray
import gc
import os
from collections.abc import Callable
import warnings

warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)


class QuantumCircuit:
    r"""
    This class allows a user to create a quantum circuit with error model from IBM machines (or an available CSV data file with the machine calibration data) where selected gates (both parameterized and non-parameterized) are implemented as methods. The gate decomposition uses the basis gates of the IBM Eagle (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)` and :math:`ECR`) / Heron (:math:`\sqrt{X}`, :math:`X`, :math:`R_Z(\theta)`, :math:`R_X(\theta)`, :math:`CZ` and :math:`RZZ(\theta)`) QPUs. Users can use custom gate decompositions/basis gates by utilizing the OpenQasm 3.0 data format to read-in quantum circuits.

    Currently, it is only possible to apply a limited selection of single and two-qubit gates to the circuit simulation (except when imported via OpenQasm). For a full list of supported gates, please refer to the Decomposition :func:`NoisyCircuits.utils.Decomposition` class documentation.

    Parameters
    -----------
    num_qubits : int
        The number of qubits in the quantum circuit.
    noise_model : dict
        A dictionary containing the raw noise model for the quantum circuit.
    backend_qpu_type : str
        The QPU architecture to use. Supported options are eagle and heron. (Defaults to heron)
    basis_gates : list[list[str]]
        A list of list of basis gates given in the format [[single qubit gates], [two qubit gates]]. (Defaults to the basis gates of the Heron QPU)
    use_fractional : bool
        A flag to determine whether to use fractional gates or not. Applicable to QPUs which allow fractional gates (Defaults to True)
    sim_backend : str
        The simulation backend to use for propogating the quantum circuit. Supported options are custom, pennylane, qiskit and qulacs. (Defaults to custom)
    threshold : float
        The threshold for pruning errors in the noise model. (Defaults to 1e-12)
    verbose : bool
        Whether to print verbose output during the noise model construction. (Defaults to True)

    Raises
    -------
    TypeError : 
        - num_qubits must be an integer.
        - noise_model must be a dictionary.
        - backend_qpu_type must be a string.
        - basis_gates must be a list of list of strings.
        - use_fractional must be a boolean.
        - sim_backend must be a string.
        - threshold must be a float.
        - verbose must be a boolean.
    ValueError :
        - num_qubits must be a positive integer.
        - backend_qpu_type must be one of the supported QPU architectures.
        - sim_backend must be one of the supported simulation backends.
        - threshold must be between 0 and 1 (exclusive).

    Notes
    -----
    - For using a different backend QPU, write in the basis gate set of the QPU in the arguement and leave the `backend_qpu_type` empty to let it default to "heron" but the customized basis gate set is given priority.
    """
    available_sim_backends = ["custom", "pennylane", "qiskit", "qulacs"]

    def __init__(
                self, 
                num_qubits:int,
                noise_model:dict,
                backend_qpu_type:str="heron",
                use_fractional:bool=True,
                sim_backend:str="custom",
                threshold:float=1e-12,
                verbose:bool=True
            )->None:
        """
        Initializes the QuantumCircuit with the specified number of qubits, noise model, number of trajectories for Monte-Carlo simulation, and threshold for noise application.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer.")
        if num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer.")
        if not isinstance(noise_model, dict):
            raise TypeError("noise_model must be a dictionary.")
        if not isinstance(backend_qpu_type, str):
            raise TypeError("backend_qpu_type must be a string.")
        if backend_qpu_type.lower() not in QuantumCircuit.basis_gates_set:
            raise ValueError(f"backend_qpu_type must be one of {list(basis_gate_set.keys())}.")
        if not isinstance(sim_backend, str):
            raise TypeError("sim_backend must be a string.")
        if sim_backend.lower() not in QuantumCircuit.available_sim_backends:
            raise ValueError(f"sim_backend must be one of {QuantumCircuit.available_sim_backends}.")
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float.")
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1 (exclusive).")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        if not isinstance(use_fractional, bool):
            raise TypeError("use_fractional must be a boolean.")
        self.num_qubits = num_qubits
        self.qpu = backend_qpu_type.lower()
        self.threshold = threshold
        self.verbose = verbose
        self._sim_backend = None
        self.solver = None
        self.sim_backend = sim_backend.lower()
        self._basis_gates = basis_gate_set[self.qpu]["basis_gates"]
        self.basis_gates = self._basis_gates
        modeller = BuildModel(
            noise_model = noise_model,
            num_qubits = self.num_qubits,
            num_cores = int(os.cpu_count() // 2),
            threshold = self.threshold,
            basis_gates = self.basis_gates,
            verbose = self.verbose
        )
        single_error, multi_error, measurement_error, connectivity = modeller.build_qubit_gate_model()
        self.single_qubit_error = {
                q : {gate : payload["qubit_channel"] for gate, payload in gates.items()} for q, gates in single_error.items()
            }
        if self.sim_backend not in ["pennylane"]:
            self.two_qubit_error = {
                gate : {pair : convert_matrix_to_little_endian(payload["qubit_channel"]) for pair, payload in pairs.items()} for gate, pairs in multi_error.items()
            }
        else:
            self.two_qubit_error = {
                gate : {pair : payload["qubit_channel"] for pair, payload in pairs.items()} for gate, pairs in multi_error.items()
            }
        self.measurement_error = measurement_error
        self.measurement_error_operator = None
        self.connectivity = connectivity
        if basis_gate_set[self.qpu]["gate_decomposition"] is not None:
            self._gate_decomposer = basis_gate_set[self.qpu]["gate_decomposition"](
                num_qubits = self.num_qubits,
                connectivity = self.connectivity,
                qubit_map = modeller.qubit_coupling_map,
                use_fractional = use_fractional
            )
        else:
            self._gate_decomposer = None
            warnings.warn("A decomposition for the given QPU does not exist and therefore, circuit building is not possible. Please import your circuit either via a OpenQasm file or as a Qiskit object.", RuntimeWarning)        
        self._ray_initialized = False

    @property
    def sim_backend(self)->str:
        """
        Getter for the _sim_backend attribute

        Returns
        --------
        str
            The name of the current simulation backend.
        """
        return self._sim_backend
    
    @sim_backend.setter
    def sim_backend(self,
                    backend:str)->None:
        """
        Setter for the _sim_backend attribute and updates the solver modules.

        Parameters
        -----------
        backend : str
            The name of the simulation backend to use. Supported options are "custom", "pennylane", "qiskit" and "qulacs".
        
        Raises
        -------
        TypeError 
            Raised when backend is not a string
        ValueError
            Raised when the specified backend is not available.
        """
        if not isinstance(backend, str):
            raise TypeError("Specified backend must of type string")
        if backend not in QuantumCircuit.available_sim_backends:
            raise ValueError(f"Specified backend {backend} is not available. Choose from {QuantumCircuit.available_sim_backends}.")
        if backend == self._sim_backend:
            print("Backend already in use.")
            return
        new_solver = load_solver(backend)
        self.solver = new_solver
        self._sim_backend = backend

    def __getattr__(self, 
                    name: str
                    ) -> Callable:
        """
        Delegate unknown attributes/methods to the selected methods class.

        Returns
        -------
        Callable
            The method corresponding to the gate name if it exists in the gate decomposer, otherwise raises an AttributeError.
        """
        if name is not None:
            return getattr(self._gate_decomposer, name)
        
    @property
    def basis_gates(self)->list[list[str]]:
        """
        Getter for the basis gates attribute.

        Returns
        -------
        list[list[str]]
            The list of basis gates supported by the simulator in the format [[single_qubit_gates], [two_qubit_gates]].
        """
        return self._basis_gates
    
    @basis_gates.setter
    def basis_gates(self,
                    basis_gates:list[list[str]]
                    )->None:
        """
        Setter for the basis gates attribute.

        Parameters
        ----------
        basis_gates : list[list[str]]
            The list of basis gates supported by the simulator in the format [[single_qubit_gates], [two_qubit_gates]].
        
        Raises
        ------
        TypeError
            If basis_gates is not a list of lists of strings.
        """
        if not isinstance(basis_gates, list) or any(not isinstance(sublist, list) for sublist in basis_gates) or any(not all(isinstance(gate, str) for gate in sublist) for sublist in self.basis_gates):
            raise TypeError("basis_gates must be a list of lists of strings.")
        self._basis_gates = basis_gates

    def refresh(self):
        """
        Resets the quantum circuit by clearing the instruction list and qubit-to-instruction mapping.
        """
        self._gate_decomposer.instruction_list = []

    def _check_gates_in_noise_model(self)->None:
        """
        Private helper method to check if the gates used in the quantum circuit are supported by the noise model. Raises a ValueError if any unsupported gates are found.

        Raises
        ------
        ValueError
            If any gates used in the quantum circuit are not supported by the noise model.k
        """
        single_qubit_gates_in_model = self.single_qubit_error[0].keys()
        two_qubit_gates_in_model = self.two_qubit_error.keys()
        gate_list = [gate for sublist in [single_qubit_gates_in_model, two_qubit_gates_in_model] for gate in sublist]
        unsupported_gate = []
        for instruction in self.instruction_list:
            if instruction[0] not in gate_list and instruction[0] != "unitary":
                unsupported_gate.append(instruction[0])
        if unsupported_gate != []:
            raise ValueError(f"The following gates are not supported by the noise model: {unsupported_gate}. Supported gates are: {gate_list}")

    def read_openqasm(self,
                    file_path:str,
                    append_to_circuit:bool=False
                    )->None:
        """
        Reads an OpenQASM 3.0 file and adds the required instructions to build the quantum circuit.

        Parameters
        ----------
        file_path : str
            The path to the OpenQASM 3.0 file.
        append_to_circuit : bool, optional
            A flag to determine whether to add the parser
        """
        try:
            parser = Parser(
                    file_path = file_path,
                    instruction_list = self._gate_decomposer.instruction_list,
                    append_to_circuit = append_to_circuit,
                    basis_gates = self.basis_gates
                )
            self._gate_decomposer.instruction_list = parser.parse()
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
        raise NotImplementedError    
    
    def _initialize_ray(self,
                        num_cores:int
                        )->None:
        """
        Initializes the Ray parallel execution environment.

        Parameters
        -----------
        num_cores : int
            The number of CPU cores to use for parallel execution.
        """
        ray.init(num_cpus=num_cores, ignore_reinit_error=True, log_to_driver=False)
        single_qubit_noise_array_ref = ray.put(np.array(list(self.single_qubit_error.items())))
        two_qubit_noise_array_ref = ray.put(np.array(list(self.two_qubit_error.items())))
        self.workers = [
            self.solver.RemoteExecutor.remote(
                num_qubits = self.num_qubits,
                single_qubit_noise = single_qubit_noise_array_ref,
                two_qubit_noise = two_qubit_noise_array_ref
            ) for _ in range(num_cores)
        ]
        self._ray_initialized = True

    def execute(self,
                qubits:list[int]=None,
                num_trajectories:int=100,
                num_cores:int=-1
                )->np.ndarray[np.float64]:
        """
        Executes the quantum circuit simulation using the Monte-Carlo Wavefunction method.

        Parameters
        -----------
        qubits : list[int], optional
            List of qubits to be measured. If None, all qubits will be measured.
        num_trajectories : int, optional
            The total number of trajectories to run. Defaults to 100.
        num_cores : int, optional
            The number of CPU cores to use for parallel execution. If -1, half of all available cores will be used. Defaults to -1.
        
        Returns
        --------
        np.ndarray[np.float64]
            An array containing the probabilities of the executed quantum circuit.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_trajectories is not an integer.
            - Raised when num_cores is not an integer.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_trajectories is not a positive integer.
            - Raised when num_cores is not a positive integer or -1.

        Notes
        -----
        This method is designed for shared memory parallel execution. In case where the RAM requirements for running a single trajectory are very high or single threaded execution is too long, users may consider the `execute_mpi` method that is specifically designed for distributed memory systems (eg. HPC clusters) for faster execution.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if not isinstance(num_trajectories, int):
            raise TypeError("num_trajectories must be an integer.")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be a positive integer.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if num_cores < 1 and num_cores != -1:
            raise ValueError("num_cores must be a positive integer or -1 for all available cores.")
        if num_cores == -1 or num_cores > os.cpu_count():
            print(f"Utilizing half of all available CPU cores: {os.cpu_count() // 2} cores.")
            num_cores = os.cpu_count() // 2
        if self.sim_backend == "custom":
            solver = self.solver.RemoteExecutor(
                num_qubits = self.num_qubits,
                single_qubit_noise = self.single_qubit_error,
                two_qubit_noise = self.two_qubit_error,
                num_cores = num_cores
            )
            probs = solver.run(
                num_trajectories = num_trajectories,
                instruction_list = self.instruction_list
            )
        else:
            if not self._ray_initialized:
                self._initialize_ray(num_cores = num_cores)
            reset_probs = [
                self.workers[i].reset.remote() for i in range(num_cores)
            ]
            futures = [
                self.workers[traj_id % num_cores].run.remote(traj_id, self.instruction_list) for traj_id in range(num_trajectories)
            ]
            prob_chunks = [
                ray.get(self.workers[i].get.remote()) for i in range(num_cores)
            ]
            probs = np.sum(prob_chunks, axis=0) / num_trajectories
        if self.sim_backend == "pennylane":
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
        if len(qubits) < self.num_qubits:
            probs = compute_marginal_probs(probs, [q for q in range(self.num_qubits) if q not in qubits])
        measurement_error_applicator.apply_measurement_error(
            probs, 
            self.measurement_error, 
            qubits, 
            len(qubits), 
            num_cores
        )
        probs = probs.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return probs

    def run_with_density_matrix(self, 
                                qubits:list[int],
                                num_cores:int=1
                                )->np.ndarray[np.float64]:
        """
        Runs the quantum circuit with the density matrix solver.

        Parameters
        -----------
        qubits : list[int]
            List of qubits to be measured.
        num_cores : int
            The number of CPU cores to use for parallel execution. Defaults to 1.

        Returns
        --------
        np.ndarray[np.float64]
            Probabilities of measuring each qubit in the computational basis.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_cores is not an integer.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_cores is not a positive integer or exceeds the number of available CPU cores.
        """
        if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
            raise TypeError("Qubits must be a list of integers.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if num_cores < 1 or num_cores > os.cpu_count():
            raise ValueError("num_cores must be a positive integer between 1 and the number of available CPU cores.")
        density_matrix_solver = self.solver.DensityMatrixSolver(
            num_qubits = self.num_qubits,
            single_qubit_noise = self.single_qubit_error,
            two_qubit_noise = self.two_qubit_error,
            instruction_list = self.instruction_list,
            num_cores = num_cores
        )
        probs = density_matrix_solver.solve(qubits=qubits)
        if self.sim_backend == "pennylane":
            probs = probs.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
        measurement_error_applicator.apply_measurement_error(
            probs, 
            self.measurement_error, 
            qubits, 
            len(qubits), 
            num_cores
        )
        probs = probs.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return probs

    def run_pure_state(self, 
                       qubits:list[int],
                       num_cores:int=1,
                       return_statevector:bool=False
                    )->np.ndarray[np.float64] | np.ndarray[np.complex128]:
        """
        Runs the quantum circuit with the pure state solver.

        Parameters
        -----------
        qubits : list[int]
            List of qubits to be measured.
        num_cores : int
            The number of CPU cores to use for parallel execution. Defaults to 1.
        return_statevector : bool
            Whether to return the final statevector instead of the probabilities. Defaults to False.

        Returns
        --------
        np.ndarray[np.float64] | np.ndarray[np.complex128]
            If return_statevector is False, returns the probabilities of measuring each qubit in the computational basis. If return_statevector is True, returns the final statevector of the quantum circuit.

        Raises
        -------
        TypeError
            - Raised when qubits is not a list of integers.
            - Raised when num_cores is not an integer.
            - Raised when return_statevector is not a boolean.
        ValueError
            - Raised when qubits contains invalid qubit indices.
            - Raised when there are no instructions in the circuit to execute.
            - Raised when num_cores is not a positive integer or exceeds the number of available CPU cores.

        Notes
        ------
        If return_statevector is set to True, the function will return the final statevector of the quantum circuit instead of the probabilities. In this case, the qubits argument will be ignored and the statevector for all qubits will be returned.
        """
        if not isinstance(return_statevector, bool):
            raise TypeError("return_statevector must be a boolean.")
        if not return_statevector:
            if not isinstance(qubits, list) or any(not isinstance(q, int) for q in qubits):
                raise TypeError("qubits must be a list of integers.")
            if any((qubit < 0 or qubit >= self.num_qubits) for qubit in qubits):
                raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        if not isinstance(num_cores, int):
            raise TypeError("num_cores must be an integer.")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to execute.")
        if num_cores < 1 or num_cores > os.cpu_count():
            raise ValueError("num_cores must be a positive integer between 1 and the number of available CPU cores.")
        pure_state_solver = self.solver.PureStateSolver(
            num_qubits = self.num_qubits,
            instruction_list = self.instruction_list,
            num_cores = num_cores,
            return_statevector = return_statevector
        )
        if return_statevector:
            output = pure_state_solver.solve()
            qubits = list(range(self.num_qubits))
        else:
            output = pure_state_solver.solve()
        if len(qubits) < self.num_qubits:
            if self._sim_backend in ["pennylane"]:
                output = output.reshape([2]*self.num_qubits).transpose(list(range(self.num_qubits))[::-1]).reshape(-1)
            output = compute_marginal_probs(output, [q for q in range(self.num_qubits) if q not in qubits])
        if self._sim_backend in ["pennylane"] and len(qubits) == self.num_qubits:
            return output
        output = output.reshape([2]*len(qubits)).transpose(list(range(len(qubits)))[::-1]).reshape(-1)
        return output

    def draw_circuit(self,
                     style:str="mpl")->None:
        """
        Draws the quantum circuit.

        Parameters
        -----------
        style : str, optional 
            The style of the drawing, either 'mpl' for matplotlib or 'text' for text-based representation. Defaults to 'mpl'.
        
        Raises
        -------
        TypeError: 
            - If style is not a string.
        ValueError: 
            - If style is not one of ['mpl', 'text'].
            - If there are no instructions in the circuit to draw.
        """
        if not isinstance(style, str):
            raise TypeError("Style must be a string.")
        if style.lower() not in ["mpl", "text"]:
            raise ValueError("Style must be one of ['mpl', 'text'].")
        if self.instruction_list == []:
            raise ValueError("No instructions in the circuit to draw.")
        from qiskit import QuantumCircuit as QiskitQuantumCircuit
        import matplotlib.pyplot as plt
        circuit = QiskitQuantumCircuit(self.num_qubits)
        instruction_map = {
            "x": lambda q, p: circuit.x(q[0]),
            "sx": lambda q, p: circuit.sx(q[0]),
            "rz": lambda q, p: circuit.rz(p, q[0]),
            "rx": lambda q, p: circuit.rx(p, q[0]),
            "cz": lambda q, p: circuit.cz(q[0], q[1]),
            "ecr": lambda q, p: circuit.ecr(q[0], q[1]),
            "rzz": lambda q, p: circuit.rzz(p, q[0], q[1]),
            "unitary": lambda q, p: circuit.unitary(p, q)
        }
        for gate_name, qubit_index, parameters in self.instruction_list:
            instruction_map[gate_name](qubit_index, parameters)
        if style.lower() == "mpl":
            circuit.draw(output="mpl")
            plt.show()
        else:
            print(circuit.draw())
        del circuit
        gc.collect()
    
    def shutdown(self):
        """
        Shutsdown the Ray parallel execution environment.
        """
        ray.shutdown()
        self._ray_initialized = False
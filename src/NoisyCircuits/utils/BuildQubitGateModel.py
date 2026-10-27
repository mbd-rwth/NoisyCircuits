# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

"""
This module allows users to generate the noise operators for single qubit and two qubit gates from the noise model dictionary obtained from the the calibration data from IBM Hardware using the class BuildModel.

The module allows users to filter out noise instructions that have a certain user-defined probability of occurrance (called threshold). The output from the module are the noise operators for single and two qubit gates, the connectivity chart for the specified device and the error operators on the measurement operation.

The connectivity chart is created from the two qubit gate instruction set where the directionality and the connection between qubits in the hardware is considered. An example usage is as follows:

Example:
    >>> from NoisyCircuits.utils.BuildQubitGateModel import BuildModel
    >>> model = BuildModel(noise_model=noise_model, num_qubits=num_qubits, num_cores=2, threshold=1e-6, basis_gates=[["sx"], ["cx"]], verbose=True)
    >>> single_qubit_noise, two_qubit_noise, connectivity, measurement_noise = model.build_qubit_gate_model()

Additionally, it needs to be noted that the QuantumCircuit module automatically builds the noise model without the user requiring to call this in the script.
"""

import numpy as np
from collections import defaultdict
import logging
from joblib import Parallel, delayed
from NoisyCircuits.utils.QubitOptimizer import select_and_relabel
from os import cpu_count

def _map_instruction_qubits(instrs, gate_qubits)->list:
    """
    Maps the qubits in the instructions to the specified gate qubits.

    Parameters
    ----------
    instrs : list
        List of instructions to map. 
    gate_qubits : dict
        Mapping of qubit indices to gate qubits.

    Returns
    -------
    list
        List of mapped instructions.
    """
    mapped = []
    for inst in instrs:
        mapped_inst = inst.copy()
        mapped_inst["qubits"] = [gate_qubits[q] for q in inst["qubits"]]
        mapped.append(mapped_inst)
    return mapped

def _extract_kraus(instrs)->list:
    """
    Extracts the Kraus operators from the instructions.

    Parameters
    ----------
    instrs : list
        List of instructions to extract Kraus operators from.

    Returns
    -------
    list
        List of Kraus operators, or None if not found.
    """
    for inst in instrs:
        if inst["name"] == "kraus":
            ops_list = [
                np.array([[complex(*e) for e in row] for row in matrix])
                for matrix in inst["params"]
            ]
            ops_list = [op.astype(np.complex128) for op in ops_list]
            return ops_list
    return None

def _ensure_ctpt(kraus_ops:list)->bool:
    """Ensures that the given Kraus operators form a CPTP map.

    Parameters
    ----------
    kraus_ops : list
        List of Kraus operators to check.

    Returns
    -------
    bool
        True if the Kraus operators form a CPTP map, False otherwise.
    
    Raises
    ------
    Warning
        If the Kraus operators do not form a CPTP map.
    """
    mat = np.zeros(kraus_ops[0].shape, dtype=np.complex128)
    for op in kraus_ops:
        mat += np.dot(op.conj().T, op)
    return np.allclose(mat, np.eye(mat.shape[0], dtype=np.complex128))

def _drop_errors(probabilities:list,
                instructions:list,
                threshold:float)->tuple[list, list]:
    """
    Filters out errors based on a probability threshold for single qubit error models.

    Parameters
    ----------
    probabilities : list
        List of error probabilities.
    instructions : list
        List of error instructions.
    threshold : float
        Probability threshold for filtering.

    Returns
    -------
    tuple[list, list]
        Filtered probabilities and instructions.
    """
    probs = np.array(probabilities)
    instrs = np.array(instructions)
    mask = probs >= threshold
    filtered_probs = probs[mask]
    filtered_instrs = instrs[mask]
    filtered_probs /= np.sum(filtered_probs)
    filtered_instrs = list(map(str, filtered_instrs))
    filtered_probs = list(map(float, filtered_probs))
    return filtered_probs, filtered_instrs

def _convert_op_to_ctpt(ops:list[np.ndarray],
                        scale:float)->list[np.ndarray]:
    """
    Helper method to apply scaling to the operators to convert them to CPTP form.

    Parameters
    ----------
    ops : list[np.ndarray]
        List of operators to scale.
    scale : float
        Scaling factor to apply, the square root of the probability.

    Returns
    -------
    list[np.ndarray]
        List of operators ensuring CPTP map.
    """
    out = [scale * op for op in ops]
    return out

def _apply_operation(op:list[np.ndarray],
                    operation_list:list[np.ndarray])->list[np.ndarray]:
    """
    Helper method to apply noise operators to a given operator.

    Parameters
    ----------
    op : list[np.ndarray]
        List of operators to apply noise to.
    operation_list : list[np.ndarray]
        List of noise operators to apply.
    
    Returns
    -------
    list[np.ndarray]
        List of resulting operators after applying noise.
    """
    ops = [np.dot(operator, o) for operator in operation_list for o in op]
    return ops
    
def _apply_local_kraus_channels(op_list:list[np.ndarray],
                                kraus_ops:list[np.ndarray])->list[np.ndarray]:
    """
    Helper method to apply local Kraus channels to a given operator.

    Parameters
    ----------
    op_list : list[np.ndarray]
        List of operators to apply noise to.
    kraus_ops : list[np.ndarray]
        List of Kraus operators to apply.
    
    Returns
    -------
    list[np.ndarray]
        List of resulting operators after applying Kraus channels.
    """
    op_list = [np.dot(kraus_op, op) for op in op_list for kraus_op in kraus_ops]
    return op_list

def _post_process_single_qubit_errors_single_qubit(qubit_errors:dict,
                                                    qubit_index:int,
                                                    num_qubits:int,
                                                    threshold:float,
                                                    instruction_map:dict,
                                                )->dict:
    """
    Method that post-processes the single-qubit errors for a single qubit on a single core.

    Parameters
    ----------
    qubit_errors : dict
        The dictionary of single-qubit errors for a specific qubit.
    qubit_index : int
        The index of the qubit being processed.
    num_qubits : int
        Total number of qubits.
    threshold : float
        Threshold for dropping errors.
    instruction_map : dict
        Map of instruction names to their matrix representations.

    Returns
    -------
    dict
        Dictionary of post-processed single-qubit errors for the specific qubit.
    """
    operator_function_map = {
        "id" : lambda x, k_ops: _apply_operation(x, [instruction_map["id"]]),
        "x" : lambda x, k_ops: _apply_operation(x, [instruction_map["x"]]),
        "y" : lambda x, k_ops: _apply_operation(x, [instruction_map["y"]]),
        "z" : lambda x, k_ops: _apply_operation(x, [instruction_map["z"]]),
        "reset" : lambda x, k_ops: _apply_operation(x, [instruction_map["K0"], instruction_map["K1"]]),
        "kraus" : lambda x, k_ops: _apply_local_kraus_channels(x, k_ops)
    }
    qubit_errors_processed = {}
    for gate in qubit_errors.keys():
        instructions = qubit_errors[gate]["instructions"]
        probabilities = qubit_errors[gate]["probabilities"]
        kraus_ops = qubit_errors[gate]["kraus"]
        if threshold is not None:
            probabilities, instructions = _drop_errors(
                                                    probabilities, 
                                                    instructions,
                                                    threshold
                                                    )
            assert len(probabilities) == len(instructions), "Probabilities and instructions must have the same length after filtering."
            assert np.allclose(np.sum(probabilities), 1), "Filtered probabilities must sum to 1."
        operators = []
        for instr, prob in zip(instructions, probabilities):
            instruction = instr.split("-")
            op = [np.eye(2, dtype=np.complex128)]
            for name in instruction:
                op = operator_function_map[name](op, kraus_ops)
            operators.extend(_convert_op_to_ctpt(op, np.sqrt(prob)))
        if not _ensure_ctpt(operators):
            print(f"Warning: Extended Kraus operators for qubit do not form a CPTP map.")
        qubit_errors_processed[gate] = {
            "qubit_channel": operators
        }
    return qubit_errors_processed

def _get_two_qubit_gate_noise_operators_single_pair(num_qubits:int, 
                                                    qpair:tuple[int, int], 
                                                    error_data:dict, 
                                                    instruction_map:dict
                                                )->dict:
    """
    Helper method to post-process the two-qubit gate errors for a single qubit pair on a single core.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits.
    qpair : tuple[int, int]
        Tuple of the two qubits in the pair.
    error_data : dict
        Dictionary containing the error data for the two-qubit gate.
    instruction_map : dict
        Map of instruction names to their matrix representations.

    Returns
    -------
    dict
        Dictionary containing the processed two-qubit gate error operators for the specific qubit pair.
    """
    full_instructions = error_data["instructions"]
    probabilities = error_data["probabilities"]
    probabilities = np.array(probabilities)
    probabilities /= np.sum(probabilities)
    probabilities = probabilities.tolist()
    kraus_ops = error_data["kraus"]
    error_operators = []
    for instruction,prob in zip(full_instructions, probabilities):
        qubits = instruction[-1]
        if len(instruction) > 2:
            major_inst = instruction[0]
            minor_inst = instruction[1]
        else:
            major_inst = "II"
            minor_inst = instruction[0]
        q0_ops = instruction_map[major_inst[0].lower()]
        q1_ops = instruction_map[major_inst[1].lower()]
        minor_ops = minor_inst.split("-")
        if "kraus" not in minor_ops:
            if "reset" not in minor_ops:
                q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                k_op = np.kron(q0_ops, q1_ops)
                error_operators.append(np.sqrt(prob) * k_op)
            else:
                if minor_ops[0] == "reset" and minor_ops[1] == "reset":
                    q0_ops1 = np.dot(instruction_map["K0"], q0_ops)
                    q0_ops2 = np.dot(instruction_map["K1"], q0_ops)
                    q1_ops1 = np.dot(instruction_map["K0"], q1_ops)
                    q1_ops2 = np.dot(instruction_map["K1"], q1_ops)
                    k_op1 = np.kron(q0_ops1, q1_ops1)
                    k_op2 = np.kron(q0_ops2, q1_ops2)
                    k_op3 = np.kron(q0_ops1, q1_ops2)
                    k_op4 = np.kron(q0_ops2, q1_ops1)
                    error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2, k_op3, k_op4])
                elif minor_ops[0] == "reset":
                    q0_op1 = np.dot(instruction_map["K0"], q0_ops)
                    q0_op2 = np.dot(instruction_map["K1"], q0_ops)
                    q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                    k_op1 = np.kron(q0_op1, q1_ops)
                    k_op2 = np.kron(q0_op2, q1_ops)
                    error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                else:
                    q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                    q1_op1 = np.dot(instruction_map["K0"], q1_ops)
                    q1_op2 = np.dot(instruction_map["K1"], q1_ops)
                    k_op1 = np.kron(q0_ops, q1_op1)
                    k_op2 = np.kron(q0_ops, q1_op2)
                    error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
        else:
            if minor_ops[0] == "kraus" and minor_ops[1] == "kraus":
                kraus_ops_for_q0 = kraus_ops[qubits[0]]
                kraus_ops_for_q1 = kraus_ops[qubits[1]]
                kraus_operations_for_q0 = []
                kraus_operations_for_q1 = []
                for kraus_op in kraus_ops_for_q0:
                    kraus_operations_for_q0.append(np.dot(kraus_op, q0_ops))
                for kraus_op in kraus_ops_for_q1:
                    kraus_operations_for_q1.append(np.dot(kraus_op, q1_ops))
                for kraus_op0 in kraus_operations_for_q0:
                    for kraus_op1 in kraus_operations_for_q1:
                        k_op = np.kron(kraus_op0, kraus_op1)
                        error_operators.append(np.sqrt(prob) * k_op)
            elif minor_ops[0] == "kraus":
                kraus_ops_for_q0 = kraus_ops[qubits[0]]
                kraus_operations_for_q0 = []
                for kraus_op in kraus_ops_for_q0:
                    kraus_operations_for_q0.append(np.dot(kraus_op, q0_ops))
                if minor_ops[1] == "reset":
                    q1_op1 = np.dot(instruction_map["K0"], q1_ops)
                    q1_op2 = np.dot(instruction_map["K1"], q1_ops)
                    for kraus_op in kraus_operations_for_q0:
                        k_op1 = np.kron(kraus_op, q1_op1)
                        k_op2 = np.kron(kraus_op, q1_op2)
                        error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                else:
                    q1_ops = np.dot(instruction_map[minor_ops[1]], q1_ops)
                    for kraus_op in kraus_operations_for_q0:
                        k_op = np.kron(kraus_op, q1_ops)
                        error_operators.append(np.sqrt(prob) * k_op)
            else:
                kraus_ops_for_q1 = kraus_ops[qubits[1]]
                kraus_operations_for_q1 = []
                for kraus_op in kraus_ops_for_q1:
                    kraus_operations_for_q1.append(np.dot(kraus_op, q1_ops))
                if minor_ops[0] == "reset":
                    q0_op1 = np.dot(instruction_map["K0"], q0_ops)
                    q0_op2 = np.dot(instruction_map["K1"], q0_ops)
                    for kraus_op in kraus_operations_for_q1:
                        k_op1 = np.kron(q0_op1, kraus_op)
                        k_op2 = np.kron(q0_op2, kraus_op)
                        error_operators.extend(np.sqrt(prob) * op for op in [k_op1, k_op2])
                else:
                    q0_ops = np.dot(instruction_map[minor_ops[0]], q0_ops)
                    for kraus_op in kraus_operations_for_q1:
                        k_op = np.kron(q0_ops, kraus_op)
                        error_operators.append(np.sqrt(prob) * k_op)
    if not _ensure_ctpt(error_operators):
        print(f"Warning: Kraus operators for qubit pair {qpair} do not form a CPTP map.")
    two_qubit_gate_error_operators = {
        "qubit_channel" : error_operators
    }
    return two_qubit_gate_error_operators


class BuildModel:
    """
    Module that generates the noise operators for each gate on each qubit from the noise model.

    Parameters
    ----------
    noise_model : dict
        The noise model to use. Provided as a JSON-ised dictionary.
    num_qubits : int
        The number of qubits in the model.
    num_cores : int, optional
        The number of cores to use for parallel processing. Defaults to 2.
    threshold : float, optional
        The threshold for noise. Defaults to 1e-12.
    basis_gates : list[list[str]]
        List of basis gates for single qubit and two qubit operators.
    verbose : bool
        Flag to indicate whether logging is required or not. Defaults to True.
               
    Raises
    ------
    TypeError
        - If noise_model is not a dictionary or num_qubits is not an integer.
        - If num_cores is not an integer.
        - If threshold is not a float or int.
        - If basis_gates is not a list of lists of strings.
        - If verbose is not a boolean.
    ValueError
        - If num_qubits is not a positive integer or threshold is not between 0 and 1.
        - If num_cores is not a positive integer, or greater than the number of available CPU cores.
        - If basis_gates is None or not a list of lists of strings.
    """
    def __init__(self,
                 noise_model : dict,
                 num_qubits : int,
                 num_cores : int=2,
                 threshold : float=1e-12,
                 basis_gates : list[list[str]]=None,
                 verbose : bool = True,
                 qubit_layout : list[int] | None = None
            ) -> None:
        """
        Initializes the BuildModel with a noise model, number of qubits, and an optional threshold.
        """
        if not isinstance(noise_model, dict):
            raise TypeError("Noise model must be a dictionary.")
        if not isinstance(num_qubits, int):
            raise TypeError("Number of qubits must be an integer.")
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        if not isinstance(num_cores, int):
            raise TypeError("Number of cores must be an integer.")
        if num_cores <= 0 or num_cores > cpu_count():
            raise ValueError("Number of cores must be a positive integer and less than or equal to the number of available CPU cores.")
        if not isinstance(threshold, (float, int)):
            raise TypeError("Threshold must be a float or int.")
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1.")
        if basis_gates is None:
            raise ValueError("Basis gates must be provided as a list of lists of strings.")
        if basis_gates and not all(isinstance(bg, list) and all(isinstance(g, str) for g in bg) for bg in basis_gates):
            raise TypeError("Basis gates must be a list of lists of strings.")
        if not isinstance(verbose, bool):
            raise TypeError("Verbose flag must be a boolean.")
        if qubit_layout is not None:
            if not isinstance(qubit_layout, list) or any(not isinstance(qubit, int) for qubit in qubit_layout):
                raise TypeError("qubit_layout must be of type list[int]")
            if any(qubit < 0 for qubit in qubit_layout):
                raise ValueError("qubits in qubit_layout must have indices greater than equal to 0.")
        self.noise_model = noise_model
        self.num_qubits = num_qubits
        self.use_qubits = qubit_layout if qubit_layout is not None else range(num_qubits)
        self.threshold = threshold
        self.basis_gates = basis_gates
        self.num_cores = num_cores
        self.qubit_coupling_map = []
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def _extract_single_qubit_qerrors(self,
                                    data:list,
                                    target_ops:list[str],
                                    allowed_qubits:list[int]
                                )->dict:
        """
        Extracts single-qubit quantum errors for the specified gates and qubits.

        Parameters
        ----------
        data : list
            List of Noise model entries from Qiskit JSON.
        target_ops : list[str]
            Gate operations to include.
        allowed_qubits : list[int]
            Qubits to include in output.

        Returns
        -------
        dict
            Nested dictionary of format: single_qubit_instructions[q][op] = {instructions, probabilities, kraus}
        """
        allowed_qubits = set(allowed_qubits)
        result = defaultdict(lambda: defaultdict(lambda: {
            "instructions" : [],
            "probabilities": [],
            "kraus" : None
        }))

        for entry in data:
            if entry.get("type") != "qerror":
                continue
            if not any(op in target_ops for op in entry.get("operations", [])):
                continue
            operations = entry["operations"]
            probabilities = entry["probabilities"]
            instruction_list = entry["instructions"]
            gate_qubits_list = entry.get("gate_qubits", [None])
            for op in operations:
                if op not in target_ops:
                    continue
                for gate_qubits in gate_qubits_list:
                    for prob, instrs in zip(probabilities, instruction_list):
                        if gate_qubits is None:
                            mapped_instrs = instrs
                            qubits = sorted(set(q for g in instrs for q in g["qubits"]))
                        else:
                            mapped_instrs = _map_instruction_qubits(instrs, gate_qubits)
                            qubits = sorted(set(q for g in mapped_instrs for q in g["qubits"]))
                        if len(qubits) != 1 or qubits[0] not in allowed_qubits:
                            continue
                        q = qubits[0]
                        instr_names = "-".join(g["name"] for g in mapped_instrs)
                        result[q][op]["instructions"].append(instr_names)
                        result[q][op]["probabilities"].append(prob)
                        if result[q][op]["kraus"] is None:
                            kraus_matrices = _extract_kraus(mapped_instrs)
                            if kraus_matrices is not None:
                                result[q][op]["kraus"] = kraus_matrices
        return dict(result)
    
    def _extract_two_qubit_errors(self,
                           data:list, 
                           allowed_qubits:list[int],
                           basis_gates:list[str])->dict:
        """
        Extracts Two Qubit errors from the noise model data.

        Parameters
        ----------
        data : list
            List of Noise model entries from Qiskit JSON.
        allowed_qubits : list[int]
            List of allowed qubits.
        basis_gates : list[str]
            List of basis gates to consider.

        Returns
        -------
        dict
            Dictionary of extracted ECR errors.
        """
        allowed_qubits = set(allowed_qubits)
        two_qubit_errors = {}
        for basis_gate in basis_gates:
            two_qubit_errors[basis_gate] = defaultdict(list)
        for entry in data:
            if entry.get("type") != "qerror":
                continue
            two_qubit_operator = entry.get("operations", [None])[0]
            if two_qubit_operator not in basis_gates:
                continue
            probabilities = entry["probabilities"]
            instructions_list = entry["instructions"]
            gate_qubits_list = entry.get("gate_qubits", [None])
            for gate_qubits in gate_qubits_list:
                if len(gate_qubits) != 2 or not set(gate_qubits).issubset(allowed_qubits):
                    continue
                self.qubit_coupling_map.append(tuple(gate_qubits))
                qpair = tuple(gate_qubits)
                for prob, instrs in zip(probabilities, instructions_list):
                    mapped_instrs = _map_instruction_qubits(instrs, gate_qubits)
                    kraus = _extract_kraus(mapped_instrs)
                    two_qubit_errors[two_qubit_operator][qpair].append({
                        "instructions" : mapped_instrs,
                        "probability" : prob,
                        "kraus" : kraus
                    })
        for basis_gate in basis_gates:
            two_qubit_errors[basis_gate] = dict(two_qubit_errors[basis_gate])
        return two_qubit_errors        
      
    def _initialize_noise_maps(self)->None:
        """
        Helper method to initialize the instruction map and operator function map for processing noise instructions.
        """
        self.instruction_map = {
            "id" : np.array([[1, 0], [0, 1]], dtype=np.complex128),
            "x" : np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "y" : np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "z" : np.array([[1, 0], [0, -1]], dtype=np.complex128),
            "K0" : np.array([[1, 0], [0, 0]], dtype=np.complex128),
            "K1" : np.array([[0, 1], [0, 0]], dtype=np.complex128),
            "i": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        }
        self.operator_function_map = {
            "id" : lambda x, k_ops: _apply_operation(x, [self.instruction_map["id"]]),
            "x" : lambda x, k_ops: _apply_operation(x, [self.instruction_map["x"]]),
            "y" : lambda x, k_ops: _apply_operation(x, [self.instruction_map["y"]]),
            "z" : lambda x, k_ops: _apply_operation(x, [self.instruction_map["z"]]),
            "reset" : lambda x, k_ops: _apply_operation(x, [self.instruction_map["K0"], self.instruction_map["K1"]]),
            "kraus" : lambda x, k_ops: _apply_local_kraus_channels(x, k_ops)
        }

    def _post_process_single_qubit_errors(self,
                                         single_qubit_errors:dict,
                                         basis_gates:list
                                        )->dict:
        """
        Post-processes the single-qubit errors to create a dictionary for qubits with their respective error instructions that can be directly applied without further processing.

        Parameters
        ----------
        single_qubit_errors : dict
            The dictionary of single-qubit errors extracted from the noise model.

        Returns
        -------
        dict
            Dictionary of post-processed single-qubit errors.
        """
        no_noise_qubit_channel = [np.eye(2, dtype=np.complex128), np.zeros((2,2), dtype=np.complex128)]
        single_qubit_errors_processed = {}
        qubits = list(self.use_qubits)
        missing = [q for q in qubits if q not in single_qubit_errors]
        if missing:
            self.logger.warning("No single qubit gate errors found for qubits {} (Treated as noiseless)".format(missing))
        qubit_error_dicts = Parallel(n_jobs = self.num_cores)(
            delayed(_post_process_single_qubit_errors_single_qubit)(
                single_qubit_errors.get(q, {}), q, self.num_qubits, self.threshold, self.instruction_map
            ) for q in qubits
        )
        for q, qubit_error_dict in zip(qubits, qubit_error_dicts):
            for gates in basis_gates:
                if gates not in qubit_error_dict:
                    qubit_error_dict[gates] = {
                        "qubit_channel" : no_noise_qubit_channel
                    }
            single_qubit_errors_processed[q] = qubit_error_dict
        return single_qubit_errors_processed

    def _post_process_two_qubit_errors(self,
                                two_qubit_errors:dict,
                                threshold:float=1e-12
                            )->dict:
        """
        Post-processes the ECR errors to create a dictionary of instructions, probabilities, and Kraus operators for each qubit pair.

        Parameters
        ----------
        two_qubit_errors : dict
            The two-qubit gate errors to process.
        threshold : float, optional
            The probability threshold for filtering errors. Defaults to 1e-12.

        Returns
        -------
        dict
            A dictionary containing the processed two-qubit errors.
        """
        processed_errors = {}
        self.logger.info("Processing two-qubit gate errors.")
        for two_qubit_gate in two_qubit_errors.keys():
            gate_error_processed = {}
            for qpair, error_list in two_qubit_errors[two_qubit_gate].items():
                instructions = []
                probabilities = []
                kraus_ops = []
                kraus_qubits = set()
                unique_kraus_per_qubit = {}  # qubit -> unique kraus operators
                filtered_count = 0
                total_count = len(error_list)
                for error_entry in error_list:
                    prob = error_entry["probability"]
                    if prob < threshold:
                        filtered_count += 1
                        continue
                    probabilities.append(prob)
                    pauli_instruction = None
                    pauli_qubits = None
                    gate_instructions = {}
                    for inst in error_entry["instructions"]:
                        inst_name = inst["name"]
                        inst_qubits = inst.get("qubits", [])
                        if inst_name == "pauli":
                            params = inst.get("params", ["II"])
                            pauli_instruction = params[0] if params else "II"
                            pauli_qubits = inst_qubits
                        elif inst_name == "kraus":
                            kraus_params = inst.get("params", [])
                            if kraus_params and inst_qubits:
                                qubit = inst_qubits[0]
                                if qubit not in unique_kraus_per_qubit:
                                    unique_kraus_per_qubit[qubit] = kraus_params
                                    kraus_qubits.add(qubit)
                            for q in inst_qubits:
                                gate_instructions[q] = inst_name
                        else:
                            for q in inst_qubits:
                                gate_instructions[q] = inst_name
                    final_instruction = []
                    qubits_list = list(qpair)
                    if pauli_instruction and pauli_qubits:
                        pauli_map = {}
                        for i, q in enumerate(pauli_qubits):
                            if i < len(pauli_instruction):
                                pauli_map[q] = pauli_instruction[i]
                        reordered_pauli = ""
                        for q in qubits_list:
                            reordered_pauli += pauli_map.get(q, "I")
                        final_instruction.append(reordered_pauli)
                    gate_parts = []
                    for q in qubits_list:
                        gate_name = gate_instructions.get(q, "id")
                        gate_parts.append(gate_name)
                    if gate_parts:
                        combined_gate = "-".join(gate_parts)
                        final_instruction.append(combined_gate)
                    final_instruction.append(qubits_list)
                    instructions.append(final_instruction)
                    kraus_ops.append(None)
                if instructions:
                    if unique_kraus_per_qubit:
                        unique_kraus_per_qubit_value = [self._extract_kraus_op(inst) for inst in unique_kraus_per_qubit.values()]
                        unique_kraus_operators = {q: ops for q, ops in zip(unique_kraus_per_qubit.keys(), unique_kraus_per_qubit_value)}
                        probabilities = np.array(probabilities)
                        probabilities /= np.sum(probabilities)
                    else:
                        unique_kraus_operators = None
                    gate_error_processed[qpair] = {
                        "instructions": instructions,
                        "probabilities": probabilities,
                        "kraus": unique_kraus_operators,
                        "kraus_qubit": list(kraus_qubits)
                    }
                    self.logger.info(f"Qubit pair {qpair}: {len(instructions)}/{total_count} errors above threshold "
                    f"({filtered_count} filtered out)")
            processed_errors[two_qubit_gate] = gate_error_processed
        self.logger.info("Two Qubit Gate errors processed.")
        return processed_errors

    def _extract_kraus_op(self,
                          kraus_op_list:list
                        )->list[np.ndarray[np.complex128]]:
        """
        Helper method to extract Kraus operators from a list of Kraus operator instructions.

        Parameters
        ----------
        kraus_op_list : list
            List of Kraus operator instructions to extract.
        
        Returns
        -------
        list[np.ndarray[np.complex128]]
            List of Kraus operators as numpy arrays of complex128 type.
        """
        ops_list = [
                np.array([[complex(*e) for e in row] for row in matrix])
                for matrix in kraus_op_list
            ]
        ops_list = [op.astype(np.complex128) for op in ops_list]
        return ops_list
        
    def _get_two_qubit_gate_noise_operators(self, 
                                           two_qubit_gate_errors:dict
                                        )->dict:
        """
        Constructs the Error Operators for the two-qubit gates from the processed error instructions.

        Parameters
        ----------
        two_qubit_gate_errors : dict
            The processed two-qubit gate errors.
        
        Returns
        -------
        dict
            Dictionary of two-qubit gate error operators.
        """
        two_qubit_gate_error_operators = {}
        use_cores = min(self.num_cores, self.num_qubits-1)
        use_cores = 1 if use_cores < 1 else use_cores
        for two_qubit_gate in two_qubit_gate_errors.keys():
            two_qubit_gate_error_operators[two_qubit_gate] = {}
            pair_operators_list = Parallel(n_jobs=use_cores)(delayed(_get_two_qubit_gate_noise_operators_single_pair)(
                                                        self.num_qubits,
                                                        qpair,
                                                        two_qubit_gate_errors[two_qubit_gate][qpair],
                                                        self.instruction_map
                                                    ) for qpair in two_qubit_gate_errors[two_qubit_gate].keys())
            for qpair, operators in zip(two_qubit_gate_errors[two_qubit_gate].keys(), pair_operators_list):
                two_qubit_gate_error_operators[two_qubit_gate][qpair] = operators                
        return two_qubit_gate_error_operators

    def _create_connectivity_map(self, 
                                 two_qubit_gate_error_instructions:dict, 
                                 use_qubits:list
                                )->dict:
        """
        Creates a connectivity map from Two Qubit Gate error instructions showing which qubits are connected.

        Parameters
        ----------
        two_qubit_gate_error_instructions : dict
            Dictionary of Two Qubit Gate error instructions with qubit pairs as keys.
        use_qubits : list
            List of qubits being used in the system.
        
        Returns
        -------
        dict
            Connectivity map showing which qubits are connected to each other.
        """
        connectivity_map = {qubit: [] for qubit in use_qubits}
        
        for qubit_pair in two_qubit_gate_error_instructions[self.basis_gates[1][0]].keys():
            if isinstance(qubit_pair, tuple) and len(qubit_pair) == 2:
                q0, q1 = qubit_pair
                if q0 in use_qubits and q1 in use_qubits:
                    if q1 not in connectivity_map[q0]:
                        connectivity_map[q0].append(q1)
                    if q0 not in connectivity_map[q1]:
                        connectivity_map[q1].append(q0)
        
        return connectivity_map

    def _extract_measurement_errors(self,
                                    data:list,
                                    extract_qubits:list[int]
                                )->dict:
        """
        Extracts the measurement errors from the noise model data.

        Parameters
        ----------
        data : list
            List of Noise model entries from Qiskit JSON.
        extract_qubits : list[int]
            List of qubits for which to extract measurement errors.
        
        Returns
        -------
        dict
            Dictionary of measurement errors for the specified qubits.
        
        Raises
        ------
        ValueError
            When a requested number of qubits exceeds the number of qubits in the system or when the requested qubits are not in the system.
        """
        measurement_errors = {}
        system_qubits = []
        for entry in data:
            if entry.get("type") != "roerror":
                continue
            operations = entry.get("operations", [])
            target_ops = ["measure"]
            if target_ops and not any(op in target_ops for op in operations):
                continue
            gate_qubits_list = entry.get("gate_qubits", [])
            system_qubits.append(gate_qubits_list[0][0])
            if gate_qubits_list[0][0] in extract_qubits:
                prob_matrix = entry.get("probabilities", [])
                measurement_errors[gate_qubits_list[0][0]] = np.array(prob_matrix).T
        if len(extract_qubits) > len(system_qubits):
            raise ValueError("Requested measurement errors for qubits that exceed the number of qubits in the system.")
        if not measurement_errors:
            self.logger.warning("No measurement errors found for the specified qubits.")
        if max(extract_qubits) > max(system_qubits):
            self.logger.warning("Requested measurement errors for qubits that that are not in the system.")
        return measurement_errors

    def build_qubit_gate_model(self)->tuple[dict, dict, dict, dict]:
        """
        Builds the qubit gate model by extracting single-qubit and two-qubit gate errors from the noise model.
        
        Returns
        -------
        tuple[dict, dict, dict, dict]
            A tuple containing the single-qubit gate error instructions, two-qubit gate error instructions, measurement error instructions and the connectivity map.

        Raises
        ------
        ValueError
            If the noise model does not contain information regarding the specified basis gates.
        """
        single_qubit_errors = self._extract_single_qubit_qerrors(
            self.noise_model["errors"],
            self.basis_gates[0],
            self.use_qubits
        )

        two_qubit_errors = self._extract_two_qubit_errors(
            self.noise_model["errors"],
            self.use_qubits,
            self.basis_gates[1]
        )
        self._initialize_noise_maps()

        if single_qubit_errors == {} and all(two_qubit_errors[gate] == {} for gate in self.basis_gates[1]):
            raise ValueError("Provided noise model does not contain information regarding the specified basis gates.")
        if all(two_qubit_errors[gate] == {} for gate in self.basis_gates[1]) and len(self.use_qubits) > 1:
            raise UserWarning("No information on two qubit gate errors were found.")
        if single_qubit_errors == {} and not all(two_qubit_errors[gate]=={} for gate in self.basis_gates[1]):
            raise UserWarning("No information on single qubit gate errors were found.")
        
        measurement_errors = self._extract_measurement_errors(self.noise_model["errors"],
                                                              self.use_qubits)

        self.logger.info("Completed Extraction of Measurement Errors.")
        self.logger.info("Completed Extraction of two-qubit gate Errors.\nStarting post-processing on Single Qubit Errors.")
        single_qubit_error_instructions = self._post_process_single_qubit_errors(single_qubit_errors, self.basis_gates[0])
        self.logger.info("Completed post-processing on Single Qubit Errors.")

        two_qubit_error_post_processed = self._post_process_two_qubit_errors(two_qubit_errors, self.threshold)
        self.logger.info("Building Noise Operators for Two Qubit Gate Errors.")

        two_qubit_gate_error_instructions = self._get_two_qubit_gate_noise_operators(two_qubit_error_post_processed)

        self.logger.info("Completed building Noise Operators for Two Qubit Gate Errors.\nExtracting Measurement Errors.")
        self.logger.info("Preparing Qubit Connectivity Map for Requested Qubits")
        connectivity_map = self._create_connectivity_map(two_qubit_gate_error_instructions, self.use_qubits)
        self.logger.info("Qubit Connectivity Map Prepared.")
        self.logger.info("Returning Single Qubit Error Instructions, Two Qubit Gate Error Instructions, Measurement Errors and Connectivity Map.")
        if self.use_qubits != list(range(self.num_qubits)):
            (single_qubit_error_instructions, two_qubit_gate_error_instructions, measurement_errors, connectivity_map, self.qubit_coupling_map) = select_and_relabel(
                single_qubit_error_instructions,
                two_qubit_gate_error_instructions,
                measurement_errors,
                connectivity_map,
                self.qubit_coupling_map,
                self.use_qubits
            )
        return single_qubit_error_instructions, two_qubit_gate_error_instructions, measurement_errors, connectivity_map
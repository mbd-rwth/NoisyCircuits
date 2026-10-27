# This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.

import numpy as np
from numba import njit
import measurement_error_applicator


def compute_marginal_probs(full_system_probs:np.ndarray[np.float64],
                           trace_qubits:list[int]
                        )->np.ndarray[np.float64]:
    """
    Function to compute the marginal probabilities by tracing out specified qubits.

    Parameters:
    -----------
    full_system_probs : np.ndarray[np.float64]
        Full system probability distribution.
    trace_qubits : list[int]
        List of qubits to be traced out.

    Returns:
    --------
    np.ndarray[np.float64]
        Marginal probability distribution after tracing out specified qubits.
    """
    n = int(np.log2(full_system_probs.shape[0]))
    probs_tensor = full_system_probs.reshape([2] * n)
    axes = [n - 1 - q for q in trace_qubits]
    probs_reduced = np.sum(probs_tensor, axis=tuple(axes))
    probs_reduced = probs_reduced.reshape(-1)
    return np.require(probs_reduced, dtype=np.float64, requirements=["C"])


def convert_matrix_to_little_endian(matrix_list:list[np.ndarray[np.complex128]])->list[np.ndarray[np.complex128]]:
    """
    Converts the two-qubit matrices from big endian to little endian format.

    Parameters:
    -----------
    matrix_list :  list[np.ndarray[np.complex128]]
         The list of input matrices in big endian format.

    Returns:
    -------
    list[np.ndarray[np.complex128]]
        The list of output matrices in little endian format.
    """
    perm = np.array([0, 2, 1, 3])
    n = matrix_list[0].shape[0]
    result_list = []
    for matrix in matrix_list:
        result = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                result[i, j] = matrix[perm[i], perm[j]]
        result_list.append(result)
    return result_list

@njit(fastmath=False)
def get_single_qubit_noise_probability(
                                gate_op:np.ndarray[np.complex128], 
                                state:np.ndarray[np.complex128], 
                                q:int
                            )->float:
    """
    Function to get the updated state of the qubit system after applying a single qubit noise operator.

    Parameters
    ----------
    gate_op : np.ndarray[np.complex128]
        The noise operator to be applied to the qubit system.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q : int
        The qubit to which the noise operator is applied.

    Returns
    -------
    float
        The probability of occurance for a Kraus Operator for the statevector - p = ⟨ψ|K|ψ⟩
    """
    dim = 1 << int(np.log2(state.size))
    stride = 1 << q
    probability = 0.0
    for p in range(0, dim >> 1, 1):
        i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1)
        j = i | stride
        n0 = gate_op[0,0] * state[i] + gate_op[0,1] * state[j]
        n1 = gate_op[1,0] * state[i] + gate_op[1,1] * state[j]
        probability += np.abs(n0)**2 + np.abs(n1)**2
    return probability

@njit(fastmath=False)
def compute_trajectory_probs_single(
                                ops:list[np.ndarray[np.complex128]], 
                                state:np.ndarray[np.complex128], 
                                qubit:int
                            )->np.ndarray[np.float64]:
    """
    Function to compute the probabilities of the noise operators for a single qubit gate.

    Parameters
    ----------
    ops : list[np.ndarray[np.complex128]]
        List of noise operators for the single qubit gate.
    state : np.ndarray[np.complex128]
        The current state of the qubit system after applying the gate.
    qubit : int
        The qubit to which the noise operators are applied.
    
    Returns
    -------
    np.ndarray[np.float64]
        The probabilities of the noise operators for the single qubit gate.
    """
    probs = np.zeros(len(ops), dtype=np.float64)
    for i in range(len(ops)):
        probs[i] = get_single_qubit_noise_probability(ops[i], state, qubit)
    return probs

@njit(fastmath=False)
def get_two_qubit_noise_probability(
                                gate_op:np.ndarray[np.complex128], 
                                state:np.ndarray[np.complex128], 
                                q1:int, 
                                q2:int
                            )->float:
    """
    Function to get the updated state of the qubit system for a two qubit noise operator.

    Parameters
    ----------
    gate_op : np.ndarray[np.complex128]
        The noise operator to be applied to the qubit system.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q1 : int
        The first qubit to which the noise operator is applied.
    q2 : int
        The second qubit to which the noise operator is applied.
    
    Returns
    -------
    float
        The probability of occurance for a Kraus Operator for the statevector - p = ⟨ψ|K|ψ⟩
    """
    probability = 0.0
    dim = 1 << int(np.log2(state.size))
    iters = dim >> 2
    q_min = min(q1, q2)
    q_max = max(q1, q2)
    m1 = (1 << q_min) - 1
    m2 = (1 << (q_max - 1)) - 1
    ull_q1 = 1 << q1
    ull_q2 = 1 << q2
    target_mask = ull_q1 | ull_q2
    for i in range(iters):
        i_s1 = (i & m1) | ((i & ~m1) << 1)
        pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1)
        idx00 = pos
        idx01 = pos | ull_q1
        idx10 = pos | ull_q2
        idx11 = pos | target_mask
        n00 = gate_op[0,0] * state[idx00] + gate_op[0,1] * state[idx01] + gate_op[0,2] * state[idx10] + gate_op[0,3] * state[idx11]
        n01 = gate_op[1,0] * state[idx00] + gate_op[1,1] * state[idx01] + gate_op[1,2] * state[idx10] + gate_op[1,3] * state[idx11]
        n10 = gate_op[2,0] * state[idx00] + gate_op[2,1] * state[idx01] + gate_op[2,2] * state[idx10] + gate_op[2,3] * state[idx11]
        n11 = gate_op[3,0] * state[idx00] + gate_op[3,1] * state[idx01] + gate_op[3,2] * state[idx10] + gate_op[3,3] * state[idx11]
        probability += np.abs(n00)**2 + np.abs(n01)**2 + np.abs(n10)**2 + np.abs(n11)**2
    return probability

@njit(fastmath=False)
def compute_trajectory_probs_two_q(ops:list[np.ndarray[np.complex128]],
                                   state:np.ndarray[np.complex128], 
                                   qubits:list[int]
                                   )->np.ndarray[np.float64]:
    """
    Function to compute the probabilities of the noise operators for a two qubit gate.

    Parameters
    ----------
    ops : list[np.ndarray[np.complex128]]
        List of noise operators for the two qubit gate.
    state : np.ndarray[np.complex128]
        The current state of the qubit system after applying the gate.
    qubits : list[int]
        The two qubits to which the noise operators are applied.
    
    Returns
    -------
    np.ndarray[np.float64]
        The probabilities of the noise operators for the two qubit gate.
    """
    probs = np.zeros(len(ops), dtype=np.float64)
    for i in range(len(ops)):
        probs[i] = get_two_qubit_noise_probability(ops[i], state, qubits[0], qubits[1])
    return probs

@njit(fastmath=False)
def update_state_inplace_1q(op:np.ndarray[np.complex128], 
                            state:np.ndarray[np.complex128], 
                            q:int
                            )->None:
    """
    Function that applies the single qubit noise operator the statevector inplace.

    Parameters
    ----------
    op : np.ndarray[np.complex128]
        The noise operator to be applied.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q : int
        The qubit to which the noise operator is applied to.
    
    Returns
    -------
    None
    """
    dim = 1 << int(np.log2(state.size))
    stride = 1 << q
    for p in range(0, dim >> 1, 1):
        i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1)
        j = i | stride
        s0 = state[i]
        s1 = state[j]
        state[i] = op[0,0]*s0 + op[0,1]*s1
        state[j] = op[1,0]*s0 + op[1,1]*s1

@njit(fastmath=False)
def update_state_inplace_2q(op:np.ndarray[np.complex128],
                            state:np.ndarray[np.complex128],
                            q1:int,
                            q2:int
                            )->None:
    """
    Function that applies the two qubit noise operator to the statevector inplace.

    Parameters
    ----------
    op : np.ndarray[np.complex128]
        The noise operator to be applied.
    state : np.ndarray[np.complex128]
        The current state of the qubit system.
    q1 : int
        The first qubit to which the noise operator is applied to.
    q2 : int
        The second qubit to which the noise operator is applied to.
    
    Returns
    -------
    None
    """
    dim = 1 << int(np.log2(state.size))
    iters = dim >> 2
    q_min = min(q1, q2)
    q_max = max(q1, q2)
    m1 = (1 << q_min) - 1
    m2 = (1 << (q_max - 1)) - 1
    ull_q1 = 1 << q1
    ull_q2 = 1 << q2
    target_mask = ull_q1 | ull_q2
    for i in range(iters):
        i_s1 = (i & m1) | ((i & ~m1) << 1)
        pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1)
        idx00 = pos
        idx01 = pos | ull_q1
        idx10 = pos | ull_q2
        idx11 = pos | target_mask
        s00 = state[idx00]
        s01 = state[idx01]
        s10 = state[idx10]
        s11 = state[idx11]
        state[idx00] = op[0,0]*s00 + op[0,1]*s01 + op[0,2]*s10 + op[0,3]*s11
        state[idx01] = op[1,0]*s00 + op[1,1]*s01 + op[1,2]*s10 + op[1,3]*s11
        state[idx10] = op[2,0]*s00 + op[2,1]*s01 + op[2,2]*s10 + op[2,3]*s11
        state[idx11] = op[3,0]*s00 + op[3,1]*s01 + op[3,2]*s10 + op[3,3]*s11

@njit(fastmath=True)
def convert_state_endianess(
                            psi:np.ndarray[np.complex128],
                            itemsize:int=8
                        ) -> None:
    """
    Function that converts the endianess of the statevector (from little to big or big to little) inplace.

    Parameters
    ----------
    psi : np.ndarray[np.complex128]
        Statevector of the quantum circuit.
    itemsize : int (Optional)
        Byte size of the numeric value. (Defaults to 8 -- np.float64/np.complex128)

    Notes
    -----
    The value of itemsize = 8 works for np.complex128 datatype because the complex128 is defined by two adjacent blocks of 8 byte float64 objects. Using this, we can swap inplace both memory-blocks (containing the real and imaginary components) in 1 go.
    """
    n = psi.shape[0] / itemsize
    for i in range(n):
        base = i * itemsize
        for j in range(itemsize // 2):
            lo = base + j
            hi = base + itemsize - 1 - j
            tmp = psi[lo]
            psi[lo] = psi[hi]
            psi[hi] = tmp

def apply_readout_error(
        prob : np.ndarray[np.float64],
        measurement_error : dict,
        qubits : list[int],
        num_threads : int
) -> np.ndarray[np.ndarray]:
    """
    Helper function to apply the measurement error on the measured qubits for noise-aware simulations.

    Parameters
    ----------
    prob : np.ndarray[np.float64]
        1D array of probabilities after noise-aware simulation
    measurement_error : dict
        Dictionary containing the stochastic measurement error matrices for each qubit in the system
    qubits : list[int]
        List of qubits that are measured
    num_threads : int
        Number of OpenMP threads for the measurement error application.

    Returns
    -------
    np.ndarray[np.float64]
        Updated probabilities post measurement error application.

    Raises
    ------
    ValueError (Raised from C++ execution)
        - Number of entries in the prob is not 2^m where m is the number of measured qubits
        - The number of error matrices in the measurement error dictionary is less than the number of measured qubits
        - The indices of the measured qubits are out of bounds.
    """
    measurement_error_structured = {
        i : measurement_error[q] for i, q in enumerate(sorted(qubits))
    }
    measurement_error_applicator.apply_measurement_error(
        prob,
        measurement_error_structured,
        list(range(len(qubits))),
        len(qubits),
        num_threads
    )
    return prob
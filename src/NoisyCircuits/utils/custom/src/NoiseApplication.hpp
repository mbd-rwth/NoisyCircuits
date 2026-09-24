/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

 /**
  * This header file consists of all necessary functions to apply noise operators for a noise-aware simulation on a shared memory program with single threaded behaviour.
  */

#pragma once
#include "TypeDefs.hpp"


// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Noise Application
// --------------------------------------------------------------------------------------------------------------------------------------


/**
 * Function that computes the probability of a Kraus operator on the state without the need for explicit buffer states for single qubit gates
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      u00 : const complex128&
 *          Reference to the (0,0) element of the Kraus Operator
 *      u01 : const complex128&
 *          Reference to the (0,1) element of the Kraus Operator
 *      u10 : const complex128&
 *          Reference to the (1,0) element of the Kraus Operator
 *      u11 : const complex128&
 *          Reference to the (1,1) element of the Kraus Operator
 *      dim : const std::size_t
 *          The total number of elements in the statevector (2^n for n qubits)
 *      stride : const std::size_t
 *          Skipping value for updating the statevector (2^q for a gate applied to qubit q)
 *      thread_count : const unsigned int
 * 
 * Returns
 *      double
 *          The probability of occurance for a Kraus Operator for the statevector - p = ⟨ψ|K|ψ⟩
 */
static inline double get_single_qubit_noise_probability(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride, cuint thread_count){
    double probability = 0.0;
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        const complex128 n0 = u00 * s0 + u01 * s1;
        const complex128 n1 = u10 * s0 + u11 * s1;
        probability += n0.real() * n0.real() + n0.imag() * n0.imag() 
                        + n1.real() * n1.real() + n1.imag() * n1.imag();
    }
    return probability;
}

/**
 * Function that applies the selected Kraus operator to the statevector inplace.
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      u00 : const complex128&
 *          Reference to the (0,0) element of the Kraus Operator
 *      u01 : const complex128&
 *          Reference to the (0,1) element of the Kraus Operator
 *      u10 : const complex128&
 *          Reference to the (1,0) element of the Kraus Operator
 *      u11 : const complex128&
 *          Reference to the (1,1) element of the Kraus Operator
 *      dim : const std::size_t
 *          The total number of elements in the statevector (2^n for n qubits)
 *      stride : const std::size_t
 *          Skipping value for updating the statevector (2^q for a gate applied to qubit q)
 *      thread_count : const unsigned int
 * 
 * Returns
 *      None
 */
static inline void apply_inplace_operator_1q(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const std::size_t dim, const std::size_t stride, cuint thread_count){
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = u00 * s0 + u01 * s1;
        state[j] = u10 * s0 + u11 * s1;
    }
}

/**
 * Function that updates the statevector with a noise operator for single qubit gates
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Index of the qubit to which the noise is applied to
 *      q_null : const std::size_t
 *          Unused - Left to ensure noise operation functions have the same function type signature.
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      noise_operators : std::vector<matrix>&
 *          Reference to the set of Kraus Operators that are to be used
 *      traj_engine : std::mt19937_64&
 *          Pre-seeded RNG Engine for random selection of a Kraus Operator
 *      thread_count : const unsigned int
 *          Number of threads to distribute tasks
 * 
 * Returns
 *      None
 */
static inline void apply_single_qubit_noise(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const std::vector<matrix>& noise_operators, std::mt19937_64& traj_engine, cuint thread_count){
    int counter = 0;
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    std::vector<double> probability_list(noise_operators.size(), 0.0);
    for (const matrix& oper : noise_operators){
        complex128 u00 = oper[0][0];
        complex128 u01 = oper[0][1];
        complex128 u10 = oper[1][0];
        complex128 u11 = oper[1][1];
        probability_list[counter] = get_single_qubit_noise_probability(state, u00, u01, u10, u11, dim, stride, thread_count);
        counter++;
    }
    std::discrete_distribution<> d(probability_list.begin(), probability_list.end());
    int c = d(traj_engine);
    auto it = std::next(noise_operators.begin(), c);
    const auto& oper = *it;
    complex128 u00 = oper[0][0];
    complex128 u01 = oper[0][1];
    complex128 u10 = oper[1][0];
    complex128 u11 = oper[1][1];
    apply_inplace_operator_1q(state, u00, u01, u10, u11, dim, stride, thread_count);
    const double p_norm = 1 / std::sqrt(probability_list[c]);
    for (std::size_t i = 0; i < dim; i++){
        state[i] *= p_norm;
    }
}

/**
 * Function that computes the probability of a Kraus operator on the state without the need for explicit buffer states for two qubit gates
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      u00 : const complex128&
 *          Reference to the (0,0) element of the Kraus Operator
 *      u01 : const complex128&
 *          Reference to the (0,1) element of the Kraus Operator
 *      u02 : const complex128&
 *          Reference to the (0,2) element of the Kraus Operator
 *      u03 : const complex128&
 *          Reference to the (0,3) element of the Kraus Operator
 *      u10 : const complex128&
 *          Reference to the (1,0) element of the Kraus Operator
 *      u11 : const complex128&
 *          Reference to the (1,1) element of the Kraus Operator
 *      u12 : const complex128&
 *          Reference to the (1,2) element of the Kraus Operator
 *      u13 : const complex128&
 *          Reference to the (1,3) element of the Kraus Operator
 *      u20 : const complex128&
 *          Reference to the (2,0) element of the Kraus Operator
 *      u21 : const complex128&
 *          Reference to the (2,1) element of the Kraus Operator
 *      u22 : const complex128&
 *          Reference to the (2,2) element of the Kraus Operator
 *      u23 : const complex128&
 *          Reference to the (2,3) element of the Kraus Operator
 *      u30 : const complex128&
 *          Reference to the (3,0) element of the Kraus Operator
 *      u31 : const complex128&
 *          Reference to the (3,1) element of the Kraus Operator
 *      u32 : const complex128&
 *          Reference to the (3,2) element of the Kraus Operator
 *      u33 : const complex128&
 *          Reference to the (3,3) element of the Kraus Operator
 *      dim : const std::size_t
 *          Total number of entries in the statevector (2^n for n qubits)
 *      iters : const std::size_t
 *          Total number of iterations required to update the statevector
 *      m1 : const std::size_t
 *          Bit-mask of all positions below the lower target qubit --> open a 0-bit gap at q_min when expanding the loop index.
 *      m2 : const std::size_t
 *          Bit-mask of all positions below the higher target qubit shifted down by 1 to account for the gap created by m1
 *      ull_q1 : const std::size_t
 *          Single-bit mask for qubit q1
 *      ull_q2 : const std::size_t
 *          Single-bit mask for qubit q1
 *      target_mask : const std::size_t
 *          Combined mask with both target-qubit bits set
 *      thread_count : const unsigned int
 *          Total number of threads to distribute tasks
 * 
 * Returns:
 *      double
 *          The probability of occurance for a Kraus Operator for the statevector - p = ⟨ψ|K|ψ⟩
 */
static inline double get_two_qubit_noise_probability(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u02, const complex128& __restrict__ u03, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const complex128& __restrict__ u12, const complex128& __restrict__ u13, const complex128& __restrict__ u20, const complex128& __restrict__ u21, const complex128& __restrict__ u22, const complex128& __restrict__ u23, const complex128& __restrict__ u30, const complex128& __restrict__ u31, const complex128& __restrict__ u32, const complex128& __restrict__ u33, const std::size_t dim, const std::size_t iters, const std::size_t m1, const std::size_t m2, const std::size_t ull_q1, const std::size_t ull_q2, const std::size_t target_mask, cuint thread_count){
    double probability = 0.0;
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const complex128 s00 = state[pos];
        const complex128 s01 = state[pos | ull_q1];
        const complex128 s10 = state[pos | ull_q2];
        const complex128 s11 = state[pos | target_mask];

        const complex128 n00 = u00 * s00 + u01 * s01 + u02 * s10 + u03 * s11;
        const complex128 n01 = u10 * s00 + u11 * s01 + u12 * s10 + u13 * s11;
        const complex128 n10 = u20 * s00 + u21 * s01 + u22 * s10 + u23 * s11;
        const complex128 n11 = u30 * s00 + u31 * s01 + u32 * s10 + u33 * s11;

        probability += n00.real() * n00.real() + n00.imag() * n00.imag()
                        + n01.real() * n01.real() + n01.imag() * n01.imag()
                        + n10.real() * n10.real() + n10.imag() * n10.imag()
                        + n11.real() * n11.real() + n11.imag() * n11.imag();
    }
    return probability;
}

/**
 * Function applies a selected Kraus operator to the statevector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      u00 : const complex128&
 *          Reference to the (0,0) element of the Kraus Operator
 *      u01 : const complex128&
 *          Reference to the (0,1) element of the Kraus Operator
 *      u02 : const complex128&
 *          Reference to the (0,2) element of the Kraus Operator
 *      u03 : const complex128&
 *          Reference to the (0,3) element of the Kraus Operator
 *      u10 : const complex128&
 *          Reference to the (1,0) element of the Kraus Operator
 *      u11 : const complex128&
 *          Reference to the (1,1) element of the Kraus Operator
 *      u12 : const complex128&
 *          Reference to the (1,2) element of the Kraus Operator
 *      u13 : const complex128&
 *          Reference to the (1,3) element of the Kraus Operator
 *      u20 : const complex128&
 *          Reference to the (2,0) element of the Kraus Operator
 *      u21 : const complex128&
 *          Reference to the (2,1) element of the Kraus Operator
 *      u22 : const complex128&
 *          Reference to the (2,2) element of the Kraus Operator
 *      u23 : const complex128&
 *          Reference to the (2,3) element of the Kraus Operator
 *      u30 : const complex128&
 *          Reference to the (3,0) element of the Kraus Operator
 *      u31 : const complex128&
 *          Reference to the (3,1) element of the Kraus Operator
 *      u32 : const complex128&
 *          Reference to the (3,2) element of the Kraus Operator
 *      u33 : const complex128&
 *          Reference to the (3,3) element of the Kraus Operator
 *      dim : const std::size_t
 *          Total number of entries in the statevector (2^n for n qubits)
 *      iters : const std::size_t
 *          Total number of iterations required to update the statevector
 *      m1 : const std::size_t
 *          Bit-mask of all positions below the lower target qubit --> open a 0-bit gap at q_min when expanding the loop index.
 *      m2 : const std::size_t
 *          Bit-mask of all positions below the higher target qubit shifted down by 1 to account for the gap created by m1
 *      ull_q1 : const std::size_t
 *          Single-bit mask for qubit q1
 *      ull_q2 : const std::size_t
 *          Single-bit mask for qubit q1
 *      target_mask : const std::size_t
 *          Combined mask with both target-qubit bits set
 *      thread_count : const unsigned int
 *          Total number of threads to distribute tasks
 * 
 * Returns:
 *      None
 */
static inline void apply_inplace_operator_2q(complex128* __restrict__ state, const complex128& __restrict__ u00, const complex128& __restrict__ u01, const complex128& __restrict__ u02, const complex128& __restrict__ u03, const complex128& __restrict__ u10, const complex128& __restrict__ u11, const complex128& __restrict__ u12, const complex128& __restrict__ u13, const complex128& __restrict__ u20, const complex128& __restrict__ u21, const complex128& __restrict__ u22, const complex128& __restrict__ u23, const complex128& __restrict__ u30, const complex128& __restrict__ u31, const complex128& __restrict__ u32, const complex128& __restrict__ u33, const std::size_t dim, const std::size_t iters, const std::size_t m1, const std::size_t m2, const std::size_t ull_q1, const std::size_t ull_q2, const std::size_t target_mask, cuint thread_count){
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx00 = pos;
        const std::size_t idx01 = pos | ull_q1;
        const std::size_t idx10 = pos | ull_q2;
        const std::size_t idx11 = pos | target_mask;

        const complex128 s00 = state[idx00];
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        const complex128 s11 = state[idx11];

        state[idx00] = u00 * s00 + u01 * s01 + u02 * s10 + u03 * s11;
        state[idx01] = u10 * s00 + u11 * s01 + u12 * s10 + u13 * s11;
        state[idx10] = u20 * s00 + u21 * s01 + u22 * s10 + u23 * s11;
        state[idx11] = u30 * s00 + u31 * s01 + u32 * s10 + u33 * s11;
    }
}

/**
 * Function to update the statevector with noise evolution after applying a unitary operation
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Qubit index 1
 *      q2 : const std::size_t
 *          Qubit index 2
 *      num_qubits : const std::size_t
 *          Total number of qubits
 *      noise_operators : const std::vector<matrix>&
 *          Kraus Operators
 *      traj_engine : std::mt19937_64&
 *          Pre-seeded RNG engine
 *      thread_count : const unsigned int
 *          Number of threads to distribute computation
 * 
 * Returns:
 *      None
 * 
 * Notes:
 *      This function is left blank as intended. Noise cannot be applied to arbitrary unitary operators acting on the circuit. This is kept to ensure consistency within the noise maps.
 */
static inline void apply_noise_for_unitary_matrix(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const std::vector<matrix>& noise_operators, std::mt19937_64& traj_engine, cuint thread_count){

}

/**
 * Function that evoles the state of the circuit with the noise operator for a two qubit gate
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Control Qubit
 *      q2 : const std::size_t
 *          Target Qubit
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      noise_operators : const std::vector<matrix>& 
 *          Reference to the set of Kraus Operators for the two qubit gate acting on qubits (q1, q2)
 *      traj_engine : std::mt19937_64&
 *          Pre-seeded RNG engine to randomly select the Kraus Operator based on the Kraus probabilities
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computations
 * 
 * Returns:
 *      None
 */
static inline void apply_two_qubit_noise(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const std::vector<matrix>& noise_operators, std::mt19937_64& traj_engine, cuint thread_count){
    const int num_operators = noise_operators.size();
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    std::vector<double> probability_list(num_operators, 0.0);
    int counter = 0;
    for (const matrix& oper : noise_operators){
        complex128 u00 = oper[0][0];
        complex128 u01 = oper[0][1];
        complex128 u02 = oper[0][2];
        complex128 u03 = oper[0][3];
        complex128 u10 = oper[1][0];
        complex128 u11 = oper[1][1];
        complex128 u12 = oper[1][2];
        complex128 u13 = oper[1][3];
        complex128 u20 = oper[2][0];
        complex128 u21 = oper[2][1];
        complex128 u22 = oper[2][2];
        complex128 u23 = oper[2][3];
        complex128 u30 = oper[3][0];
        complex128 u31 = oper[3][1];
        complex128 u32 = oper[3][2];
        complex128 u33 = oper[3][3];
        probability_list[counter] = get_two_qubit_noise_probability(state, u00, u01, u02, u03, u10, u11, u12, u13, u20, u21, u22, u23, u30, u31, u32, u33, dim, iters, m1, m2, ull_q1, ull_q2, target_mask, thread_count);
        counter++;
    }
    std::discrete_distribution<> d(probability_list.begin(), probability_list.end());
    int c = d(traj_engine);
    auto it = std::next(noise_operators.begin(), c);
    const auto& oper = *it;
    complex128 u00 = oper[0][0];
    complex128 u01 = oper[0][1];
    complex128 u02 = oper[0][2];
    complex128 u03 = oper[0][3];
    complex128 u10 = oper[1][0];
    complex128 u11 = oper[1][1];
    complex128 u12 = oper[1][2];
    complex128 u13 = oper[1][3];
    complex128 u20 = oper[2][0];
    complex128 u21 = oper[2][1];
    complex128 u22 = oper[2][2];
    complex128 u23 = oper[2][3];
    complex128 u30 = oper[3][0];
    complex128 u31 = oper[3][1];
    complex128 u32 = oper[3][2];
    complex128 u33 = oper[3][3];
    apply_inplace_operator_2q(state, u00, u01, u02, u03, u10, u11, u12, u13, u20, u21, u22, u23, u30, u31, u32, u33, dim, iters, m1, m2, ull_q1, ull_q2, target_mask, thread_count);
    const double p_norm = 1 / std::sqrt(probability_list[c]);
    for (std::size_t i = 0; i < dim; i++){
        state[i] *= p_norm;
    }
}
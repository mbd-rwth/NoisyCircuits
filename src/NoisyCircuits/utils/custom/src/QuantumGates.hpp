/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This header file consists of all necessary functions to apply quantum gates to a statevector. 
 * See the article: Díaz,  G. J., Steffenel,  L. A., Barrios,  C. J., & Couturier,  J. F. (2024). How to Build a Software Quantum Simulator. Preprints. https://doi.org/10.20944/preprints202409.1497.v1 for more information on the logic of applying local gate matrices to an entire statevector.
 */

#pragma once
#include "TypeDefs.hpp"

// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Quantum Gate Application
// --------------------------------------------------------------------------------------------------------------------------------------

/**
 * Function that applies the X (NOT) Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_X_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = s1;
        state[j] = s0;
    }
}

/**
 * Function that applies the RZ Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Angle of rotation
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_RZ_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = complex128(cosine * s0.real() + sine * s0.imag(), cosine * s0.imag() - sine * s0.real());
        state[j] = complex128(cosine * s1.real() - sine * s1.imag(), cosine * s1.imag() + sine * s1.real());
    }
}

/**
 * Function that applies the RX Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Angle of rotation
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_RX_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = complex128(cosine * s0.real() + sine * s1.imag(), cosine * s0.imag() - sine * s1.real());
        state[j] = complex128(sine * s0.imag() + cosine * s1.real(), cosine * s1.imag() - sine * s0.real());
    }
}

/**
 * Function that applies the SX Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_SX_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    constexpr complex128 post_i = complex128(0.5, 0.5);
    constexpr complex128 negt_i = complex128(0.5, -0.5);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = post_i * s0 + negt_i * s1;
        state[j] = negt_i * s0 + post_i * s1;
    }
}

/**
 * Function that applies the RY Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Angle of rotation
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_RY_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = complex128(cosine * s0.real() - sine * s1.real(), cosine * s0.imag() - sine * s1.imag());
        state[j] = complex128(sine * s0.real() + cosine * s1.real(), sine * s0.imag() + cosine * s1.imag());
    }
}

/**
 * Function that applies the Hadamard Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_H_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    constexpr double inv_sqrt_2 = 0.7071067811865475;
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = inv_sqrt_2 * (s0 + s1);
        state[j] = inv_sqrt_2 * (s0 - s1);
    }
}

/**
 * Function that applies the Phase Gate to the qubit q and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q : const std::size_t
 *          Qubit index to which to apply the gate
 *      q_null : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Phase Shift angle
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_P_gate(complex128* __restrict__ state, const std::size_t q, const std::size_t q_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << q;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t pair = 0; pair < (dim >> 1); ++pair){
        const std::size_t i = (pair & (stride - 1)) | ((pair & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const complex128 s0 = state[i];
        const complex128 s1 = state[j];
        state[i] = s0;
        state[j] = complex128(s1.real() * cosine - s1.imag() * sine, s1.real() * sine + s1.imag() * cosine);
    }
}

/**
 * Function that applies the CZ Gate to the qubit q2 with qubit q1 as the control and modifies the state vector inplace
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
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_CZ_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t m1 = (1ULL << (q1 < q2 ? q1 : q2)) - 1;
    const std::size_t m2 = (1ULL << ((q1 > q2 ? q1 : q2) - 1)) - 1;
    const std::size_t target_mask = (1ULL << q1) | (1ULL << q2);
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t i = 0; i < iters; ++i){
        const size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        state[pos | target_mask] = -state[pos | target_mask];
    }
}

/**
 * Function that applies the RZZ Gate to the qubits q1 and q2, and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Qubit Index 1
 *      q2 : const std::size_t
 *          Qubit Index 2
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Angle of rotation
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_RZZ_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    #pragma omp parallel for num_threads(thread_count)
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
        state[idx00] = complex128(cosine * s00.real() + sine * s00.imag(), cosine * s00.imag() - sine * s00.real());
        state[idx01] = complex128(cosine * s01.real() - sine * s01.imag(), cosine * s01.imag() + sine * s01.real());
        state[idx10] = complex128(cosine * s10.real() - sine * s10.imag(), cosine * s10.imag() + sine * s10.real());
        state[idx11] = complex128(cosine * s11.real() + sine * s11.imag(), cosine * s11.imag() - sine * s11.real());
    }
}

/**
 * Function that applies the ECR Gate to the qubit q2 with qubit q1 as the control and modifies the state vector inplace
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
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_ECR_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    constexpr double inv_sqrt_2 = 0.7071067811865476;
    constexpr complex128 inv_sqrt_2i = complex128(0.0, 0.7071067811865476);
    #pragma omp parallel for num_threads(thread_count)
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
        state[idx00] = inv_sqrt_2 * s01 + inv_sqrt_2i * s11;
        state[idx01] = inv_sqrt_2 * s00 - inv_sqrt_2i * s10;
        state[idx10] = inv_sqrt_2 * s11 + inv_sqrt_2i * s01;
        state[idx11] = inv_sqrt_2 * s10 - inv_sqrt_2i * s00;
    }
}

/**
 * Function that applies the CX Gate to the qubit q2 with qubit q1 as the control and modifies the state vector inplace
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
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_CX_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx01 = pos | ull_q1;
        const std::size_t idx11 = pos | target_mask;
        const complex128 s01 = state[idx01];
        const complex128 s11 = state[idx11];
        state[idx01] = s11;
        state[idx11] = s01;
    }
}

/**
 * Function that applies the SWAP Gate to qubits q1 and q2 and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Qubit Index 1
 *      q2 : const std::size_t
 *          Qubit Index 2
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          Unused variable, kept for type signature consistency
 *      target_qubits : const std::vector<std::size_t>& 
 *          Unused variable, kept for type signature consistency
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_SWAP_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t iters = dim >> 2;
    const std::size_t q_min = q1 < q2 ? q1 : q2;
    const std::size_t q_max = q1 > q2 ? q1 : q2;
    const std::size_t m1 = (1ULL << q_min) - 1;
    const std::size_t m2 = (1ULL << (q_max - 1)) - 1;
    const std::size_t ull_q1 = 1ULL << q1;
    const std::size_t ull_q2 = 1ULL << q2;
    const std::size_t target_mask = ull_q1 | ull_q2;
    #pragma omp parallel for num_threads(thread_count)
    for (std::size_t i = 0; i < iters; ++i){
        const std::size_t i_s1 = (i & m1) | ((i & ~m1) << 1);
        const std::size_t pos = (i_s1 & m2) | ((i_s1 & ~m2) << 1);
        const std::size_t idx01 = pos | ull_q1;
        const std::size_t idx10 = pos | ull_q2;
        const complex128 s01 = state[idx01];
        const complex128 s10 = state[idx10];
        state[idx01] = s10;
        state[idx10] = s01;
    }
}

/**
 * Function that applies an arbitrary unitary operator and modifies the state vector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      q2 : const std::size_t
 *          Unused variable, kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable, kept for type signature consistency
 *      U : const matrix&
 *          The unitary operator that is to be applied to the circuit
 *      target_qubits : const std::vector<std::size_t>& 
 *          Set of qubits to which the unitary operator is to be applied to, must be in ascending order
 *      thread_count : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_unitary_gate(complex128* __restrict__ state, const std::size_t q_null, const std::size_t q_null2, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t num_targets = target_qubits.size();
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t inner_dim = std::size_t{1} << num_targets;
    const std::size_t outer_dim = dim >> num_targets;
    std::vector<unsigned char> is_target(num_qubits, 0);
    for (std::size_t t = 0; t < num_targets; ++t){
        is_target[target_qubits[t]] = 1;
    }
    std::vector<std::size_t> target_offsets(inner_dim, 0);
    for (std::size_t inner = 0; inner < inner_dim; ++inner){
        std::size_t idx = 0;
        for (std::size_t b = 0; b < num_targets; ++b){
            if (inner & (std::size_t{1} << b)) {
                idx |= (std::size_t{1} << target_qubits[b]);
            }
        }
        target_offsets[inner] = idx;
    }

    if (outer_dim >= inner_dim){
        #pragma omp parallel num_threads(thread_count)
        {
            std::vector<complex128> temp(inner_dim);
            std::vector<complex128> new_local(inner_dim);

            #pragma omp for schedule(static)
            for (std::size_t outer = 0; outer < outer_dim; ++outer){
                std::size_t base_idx = 0;
                std::size_t src_idx = outer;
                for (std::size_t bit = 0; bit < num_qubits; ++bit){
                    if (is_target[bit]){
                        continue;
                    }
                    if  (src_idx & 1ULL){
                        base_idx |= (std::size_t{1} << bit);
                    }
                    src_idx >>= 1;
                }
                for (std::size_t i = 0; i < inner_dim; ++i){
                    temp[i] = state[base_idx | target_offsets[i]];
                }
                for (std::size_t r = 0; r < inner_dim; ++r){
                    complex128 sum(0.0, 0.0);
                    for (std::size_t c = 0; c < inner_dim; ++c){
                        sum += U[r][c] * temp[c];
                    }
                    new_local[r] = sum;
                }
                for (std::size_t i = 0; i < inner_dim; ++i){
                    state[base_idx | target_offsets[i]] = new_local[i];
                }
            }
        }

    }
    else {
        for (std::size_t outer = 0; outer < outer_dim; ++outer){
            std::size_t base_idx = 0;
            std::size_t src_idx = outer;
            for (std::size_t bit = 0; bit < num_qubits; ++bit){
                if (is_target[bit]){
                    continue;
                }
                if (src_idx & 1ULL){
                    base_idx |= (std::size_t{1} << bit);
                }
                src_idx >>= 1;
            }

            std::vector<complex128> temp(inner_dim);
            std::vector<complex128> new_local(inner_dim);

            #pragma omp parallel num_threads(thread_count)
            {
                #pragma omp for schedule(static)
                for (std::size_t i = 0; i < inner_dim; ++i){
                    temp[i] = state[base_idx | target_offsets[i]];
                }
                
                #pragma omp for schedule(static)
                for (std::size_t r = 0; r < inner_dim; ++r){
                    complex128 sum(0.0, 0.0);
                    const auto& row = U[r];
                    for (std::size_t c = 0; c < inner_dim; ++c){
                        sum += row[c] * temp[c];
                    }
                    new_local[r] = sum;
                }
                #pragma omp for schedule(static)
                for (std::size_t i = 0; i < inner_dim; ++i){
                    state[base_idx | target_offsets[i]] = new_local[i];
                }
            }
        }
    }
}
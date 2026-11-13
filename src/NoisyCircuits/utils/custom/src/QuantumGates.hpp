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
#include <algorithm>
#include <cmath>

// Set the minimum number of amplitude pairs before allowing parallel execution.
constexpr std::size_t MIN_PARALLEL_GROUPS = std::size_t{1} << 14;

// --------------------------------------------------------------------------------------------------------------------------------------
// Define cell kernels for loop vectorization for singe and two qubit gates 
// --------------------------------------------------------------------------------------------------------------------------------------
template <typename Kernel>
static inline void for_each_pair(complex128* __restrict__ state, const std::size_t dim, const std::size_t stride, cuint thread_count, Kernel kernel){
    const std::size_t pairs = dim >> 1;
    #pragma omp parallel num_threads(thread_count) if(pairs >=  MIN_PARALLEL_GROUPS)
    {
        const std::size_t num_threads = omp_get_num_threads();
        const std::size_t thread_number = omp_get_thread_num();
        std::size_t p = pairs * thread_number / num_threads;
        const std::size_t end_pair = pairs * (thread_number + 1) / num_threads;
        if (stride >= 4){
            while (p < end_pair){
                const std::size_t end_run = std::min(end_pair, (p | (stride - 1)) + 1);
                const std::size_t i0 = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1);
                const std::size_t length = end_run - p;
                #pragma omp simd
                for (std::size_t k = 0; k < length; ++k){
                    kernel(state, i0 + k, i0 + k + stride);
                }
                p = end_run;
            }
        }
        else {
            for (; p < end_pair; ++p){
                const std::size_t i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1);
                kernel(state, i, i + stride);
            }
        }
    }
}

template <typename Kernel>
static inline void for_each_quad(complex128* __restrict__ state, const std::size_t dim, const std::size_t s1, const std::size_t s2, cuint thread_count, Kernel kernel){
    const std::size_t quads = dim >> 2;
    const std::size_t low = std::min(s1, s2);
    const std::size_t high = std::max(s1, s2);
    #pragma omp parallel num_threads(thread_count) if(quads >= MIN_PARALLEL_GROUPS)
    {
        const std::size_t num_threads = omp_get_num_threads();
        const std::size_t thread_number = omp_get_thread_num();
        std::size_t g = quads * thread_number / num_threads;
        const std::size_t end_g = quads * (thread_number + 1) / num_threads;
        if (low >= 4){
            while (g < end_g){
                const std::size_t end_run = std::min(end_g, (g | (low - 1)) + 1);
                const std::size_t g_low = (g & (low - 1)) | ((g & ~(low - 1)) << 1);
                const std::size_t i0 = (g_low & (high - 1)) | ((g_low & ~(high - 1)) << 1);
                const std::size_t length = end_run - g;
                # pragma omp simd
                for (std::size_t k = 0; k < length; ++k){
                    kernel(state, i0 + k);
                }
                g = end_run;
            }
        }
        else {
            for (; g < end_g; ++g){
                const std::size_t g_low = (g & (low - 1)) | ((g & ~(low - 1)) << 1);
                kernel(state, (g_low & (high - 1)) | ((g_low & ~(high - 1)) << 1));
            }
        }
    }
}

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
    for_each_pair(state, dim, stride, thread_count, [](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const complex128 s0 = s[i];
        s[i] = s[j];
        s[j] = s0;
    });
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
    for_each_pair(state, dim, stride, thread_count, [cosine, sine](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(cosine * a + sine * b, cosine * b - sine * a);
        s[j] = complex128(cosine * c - sine * d, cosine * d + sine * c);
    });
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
    for_each_pair(state, dim, stride, thread_count, [cosine, sine](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(cosine * a + sine * d, cosine * b - sine * c);
        s[j] = complex128(sine * b + cosine * c, cosine * d - sine * a);
    });
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
    for_each_pair(state, dim, stride, thread_count, [](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(0.5 * (a - b + c + d), 0.5 * (a + b - c + d));
        s[j] = complex128(0.5 * (a + b + c - d), 0.5 * (-a + b + c + d));
    });
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
    for_each_pair(state, dim, stride, thread_count, [cosine, sine](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(cosine * a - sine * c, cosine * b - sine * d);
        s[j] = complex128(sine * a + cosine * c, sine * b + cosine * d);
    });
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
    for_each_pair(state, dim, stride, thread_count, [](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(inv_sqrt_2 * (a + c), inv_sqrt_2 * (b + d));
        s[j] = complex128(inv_sqrt_2 * (a - c), inv_sqrt_2 * (b - d));
    });
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
    for_each_pair(state, dim, stride, thread_count, [cosine, sine](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double c = s[j].real(), d = s[j].imag();
        s[j] = complex128(c * cosine - d * sine, c * sine + d * cosine);
    });
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
    const std::size_t s1 = std::size_t{1} << q1;
    const std::size_t s2 = std::size_t{1} << q2;
    const std::size_t s11 = s1 + s2;
    for_each_quad(state, dim, s1, s2, thread_count, [s11](complex128* __restrict__ s, const std::size_t i){
        s[i + s11] = -s[i + s11];
    });
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
    const std::size_t sq1 = std::size_t{1} << q1;
    const std::size_t sq2 = std::size_t{1} << q2;
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    for_each_quad(state, dim, sq1, sq2, thread_count, [sq1, sq2, cosine, sine](complex128* __restrict__ s, const std::size_t i){
        const std::size_t idx[4] = {i, i + sq1, i + sq2, i + sq1 + sq2};
        const double sign[4] = {1.0, -1.0, -1.0, 1.0};
        for (int k = 0; k < 4; ++k){
            const double a = s[idx[k]].real(), b = s[idx[k]].imag();
            s[idx[k]] = complex128(cosine * a + sign[k] * sine * b, cosine * b - sign[k] * sine * a);
        }
    });
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
    const std::size_t sq1 = std::size_t{1} << q1;
    const std::size_t sq2 = std::size_t{1} << q2;
    constexpr double r = 0.7071067811865475;
    for_each_quad(state, dim, sq1, sq2, thread_count, [sq1, sq2](complex128* __restrict__ s, const std::size_t i){
        const std::size_t i01 = i + sq1, i10 = i + sq2, i11 = i + sq1 + sq2;
        const double a00 = s[i].real(), b00 = s[i].imag(), a01 = s[i01].real(), b01 = s[i01].imag();
        const double a10 = s[i10].real(), b10 = s[i10].imag(), a11 = s[i11].real(), b11 = s[i11].imag();
        s[i] = complex128(r * (a01 - b11), r * (b01 + a11));
        s[i01] = complex128(r * (a00 + b10), r * (b00 - a10));
        s[i10] = complex128(r * (a11 - b01), r * (b11 + a01));
        s[i11] = complex128(r * (a10 + b00), r * (b10 - a00));
    });
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
    const std::size_t sq1 = std::size_t{1} << q1;
    const std::size_t sq2 = std::size_t{1} << q2;
    for_each_quad(state, dim, sq1, sq2, thread_count, [sq1, sq2](complex128* __restrict__ s, const std::size_t i){
        const complex128 s01 = s[i + sq1];
        s[i + sq1] = s[i + sq1 + sq2];
        s[i + sq1 + sq2] = s01;
    });
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
    const std::size_t sq1 = std::size_t{1} << q1;
    const std::size_t sq2 = std::size_t{1} << q2;
    for_each_quad(state, dim, sq1, sq2, thread_count, [sq1, sq2](complex128* __restrict__ s, const std::size_t i){
        const complex128 s01 = s[i + sq1];
        s[i + sq1] = s[i + sq2];
        s[i + sq2] = s01;
    });
}

// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Applying Arbitrary Quantum Unitary Gates
// --------------------------------------------------------------------------------------------------------------------------------------

/**
 * Function that insert a zero bit at each of the bit positions given by the index
 * 
 * Inputs:
 *      index : std::size_t
 *          Index into which the zero bits are inserted
 *      sorted_targets : const std::size_t*
 *          Pointer to the bit positions at which to insert the zero bits, given in ascending order
 *      num_targets : const std::size_t
 *          Total number of bit positions (number of qubits in the system)
 * 
 * Returns:
 *      std::size_t
 *          Index with a zero bit at every position in the sorted targets
 */
static inline std::size_t insert_zero_bits(std::size_t index, const std::size_t* sorted_targets, const std::size_t num_qubits){
    for (std::size_t i = 0; i < num_qubits; ++i){
        const std::size_t low = (std::size_t{1} << sorted_targets[i]) - 1;
        index = (index & low) | ((index & ~low) << 1);
    }
    return index;
}

/**
 * Function that applies a diagonal unitary matrix to a given state inplace
 * 
 * Inputs: 
 *      state : complex128*
 *          Pointer to the statevector of the system
 *      dim : const std::size_t
 *          Length of the statevector
 *      U : const matrix&
 *          Reference to the diagonal unitary matrix
 *      target_qubits : std::vector<std::size_t>&
 *          Reference to the vector containing the indices of the target qubits
 *      thread_count : const unsigned int
 *          Number of threads to use for parallelization
 * 
 * Returns:
 *      None
 */
static inline void apply_diagonal_matrix(complex128* __restrict__ state, const std::size_t dim, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t num_target_qubits = target_qubits.size();
    const std::size_t inner_dim = std::size_t{1} << num_target_qubits;
    std::vector<double> diagonal_real(inner_dim), diagonal_imag(inner_dim);
    for (std::size_t i = 0; i < inner_dim; ++i){
        diagonal_imag[i] = U[i][i].imag();
        diagonal_real[i] = U[i][i].real();
    }
    const std::size_t smallest_target = *std::min_element(target_qubits.begin(), target_qubits.end());
    const std::size_t block_size = std::size_t{1} << std::min<std::size_t>(smallest_target, 12);
    const std::size_t num_blocks = dim / block_size;
    const double* __restrict__ d_real = diagonal_real.data();
    const double* __restrict__ d_imag = diagonal_imag.data();

    #pragma omp parallel for schedule(static) num_threads(thread_count) if(dim >= (MIN_PARALLEL_GROUPS << 1))
    for (std::size_t block = 0; block < num_blocks; ++block){
        const std::size_t base = block * block_size;
        std::size_t l = 0;
        for (std::size_t b = 0; b < num_target_qubits; ++b){
            l |= ((base >> target_qubits[b]) & 1) << b;
        }
        const double phase_real = d_real[l], phase_imag = d_imag[l];
        #pragma omp simd
        for (std::size_t i = base; i < base + block_size; ++i){
            const double a = state[i].real(), b = state[i].imag();
            state[i] = complex128(phase_real * a - phase_imag * b, phase_real * b + phase_imag * a);
        }
    }
}


/**
 * Function that applies a dense unitary operator on a fixed numbber of qubits and updates the state inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      dim : const std::size_t
 *          Lenght of the statevector
 *      U : const matrix&
 *          Reference to the dense unitary matrix
 *      target_qubits : const std::vector<std::size_t>&
 *          Reference to the vector containing the indices of the target qubits
 *      thread_count : const unsigned int
 *          Number of threads to use for parallelization
 * 
 * Returns:
 *      None
 */
template <std::size_t K>
static inline void apply_dense_unitary_fixed(complex128* __restrict__ state, const std::size_t dim, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    constexpr std::size_t D = std::size_t{1} << K;
    double U_re[D][D], U_im[D][D];
    for (std::size_t r = 0; r < D; ++r){
        for (std::size_t c = 0; c < D; ++c){
            U_re[r][c] = U[r][c].real();
            U_im[r][c] = U[r][c].imag();
        }
    }
    std::size_t offsets[D];
    for (std::size_t l = 0; l < D; ++l){
        offsets[l] = 0;
        for (std::size_t b = 0; b < K; ++b){
            if ((l >> b) & 1) offsets[l] |= std::size_t{1} << target_qubits[b];
        }
    }
    std::size_t sorted_targets[K];
    std::copy(target_qubits.begin(), target_qubits.end(), sorted_targets);
    std::sort(sorted_targets, sorted_targets + K);
    const std::size_t groups = dim >> K;

    #pragma omp parallel for schedule(static) num_threads(thread_count) if(groups >= MIN_PARALLEL_GROUPS)
    for (std::size_t g = 0; g < groups; ++g){
        const std::size_t base = insert_zero_bits(g, sorted_targets, K);
        double v_real[D], v_imag[D];
        for (std::size_t c = 0; c < D; ++c){
            v_real[c] = state[base + offsets[c]].real();
            v_imag[c] = state[base + offsets[c]].imag();
        }
        for (std::size_t r = 0; r < D; ++r){
            double sum_real = 0.0, sum_imag = 0.0;
            for (std::size_t c = 0; c < D; ++c){
                sum_real += U_re[r][c] * v_real[c] - U_im[r][c] * v_imag[c];
                sum_imag += U_re[r][c] * v_imag[c] + U_im[r][c] * v_real[c];
            }
            state[base + offsets[r]] = complex128(sum_real, sum_imag);
        }
    }
}

/**
 * Function that applies a dense unitary operator on any number of qubits and updates the statevector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      dim : const std::size_t
 *          Length of the statevector
 *      U : const matrix&
 *          Reference to the dense unitary operator
 *      target_qubits : const std::vector<std::size_t>&
 *          Reference to the vector containing the indices to the target qubits
 *      thread_count : const unsigned int
 *          Number of threads to use for parallelization
 * 
 * Returns:
 *      None
 */
static inline void apply_dense_unitary_generic(complex128* __restrict__ state, const std::size_t dim, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t num_targets = target_qubits.size();
    const std::size_t D = std::size_t{1} << num_targets;
    std::vector<double> U_re(D * D), U_im(D * D);
    for (std::size_t r = 0; r < D; ++r){
        for (std::size_t c = 0; c < D; ++c){
            U_re[r * D + c] = U[r][c].real();
            U_im[r * D + c] = U[r][c].imag();
        }
    }
    std::vector<std::size_t> offsets(D, 0);
    for (std::size_t l = 0; l < D; ++l){
        for (std::size_t b = 0; b < num_targets; ++b){
            if ((l >> b) & 1) offsets[l] |= std::size_t{1} << target_qubits[b];
        }
    }
    std::vector<std::size_t> sorted_targets(target_qubits.begin(), target_qubits.end());
    std::sort(sorted_targets.begin(), sorted_targets.end());
    const std::size_t groups = dim >> num_targets;

    #pragma omp parallel num_threads(thread_count) if(groups >= MIN_PARALLEL_GROUPS)
    {
        std::vector<double> v_real(D), v_imag(D);
        #pragma omp for schedule(static)
        for (std::size_t g = 0; g < groups; ++g){
            const std::size_t base = insert_zero_bits(g, sorted_targets.data(), num_targets);
            for (std::size_t c = 0; c < D; ++c){
                v_real[c] = state[base + offsets[c]].real();
                v_imag[c] = state[base + offsets[c]].imag();
            }
            for (std::size_t r = 0; r < D; ++r){
                const double* __restrict__ row_real = &U_re[r * D];
                const double* __restrict__ row_imag = &U_im[r * D];
                double sum_real = 0.0, sum_imag = 0.0;
                #pragma omp simd reduction(+:sum_real, sum_imag)
                for (std::size_t c = 0; c < D; ++c){
                    sum_real += row_real[c] * v_real[c] - row_imag[c] * v_imag[c];
                    sum_imag += row_real[c] * v_imag[c] + row_imag[c] * v_real[c];
                }
                state[base + offsets[r]] = complex128(sum_real, sum_imag);
            }
        }
    }
}

/**
 * Function that applies an arbitrary unitary operator and updates the state inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the system statevector
 *      q1_null : const std::size_t
 *          Unused variable and kept for type signature consistency
 *      q2_null : const std::size_t
 *          Unused variable and kept for type signature consistency
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      theta : const double
 *          Unused variable and kept for type signature consistency
 *      U : const matrix&
 *          Reference to the unitary operator matrix
 *      target_qubits : const std::vector<std::size_t>&
 *          Reference to the vector containing the indices of the target qubits
 *      thread_count : const unsigned int
 *          Number of threads for parallelization
 * 
 * Returns:
 *      None
 */
static inline void apply_unitary_gate(complex128* __restrict__ state, const std::size_t q1_null, const std::size_t q2_null, const std::size_t num_qubits, const double theta, const matrix& U, const std::vector<std::size_t>& target_qubits, cuint thread_count){
    const std::size_t num_targets = target_qubits.size();
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t inner_dim = std::size_t{1} << num_targets;

    // Skip identity operators
    bool diagonal = true, identity = true;
    for (std::size_t r = 0; r < inner_dim && diagonal; ++r){
        for (std::size_t c = 0; c < inner_dim; ++c){
            if (r != c && U[r][c] != 0.0){
                diagonal = false;
                identity = false;
                break;
            }
            if (r == c && U[r][c] != 1.0){
                identity = false;
            }
        }
    }
    if (identity){
        return;
    }
    if (diagonal){
        apply_diagonal_matrix(state, dim, U, target_qubits, thread_count);
        return;
    }

    // Apply specific implementations for 1 and 2 target qubits
    if (num_targets == 1){
        const double u00_real = U[0][0].real(), u01_real = U[0][1].real(), u10_real = U[1][0].real(), u11_real = U[1][1].real(); 
        const double u00_imag = U[0][0].imag(), u01_imag = U[0][1].imag(), u10_imag = U[1][0].imag(), u11_imag = U[1][1].imag();
        for_each_pair(state, dim, std::size_t{1} << target_qubits[0], thread_count, [=](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
            const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
            s[i] = complex128(u00_real * a - u00_imag * b + u01_real * c - u01_imag * d, u00_real * b + u00_imag * a + u01_real * d + u01_imag * c);
            s[j] = complex128(u10_real * a - u10_imag * b + u11_real * c - u11_imag * d, u10_real * b + u10_imag * a + u11_real * d + u11_imag * c);
        });
    }
    else if (num_targets == 2){
        double U_re[4][4], U_im[4][4];
        for (int r = 0; r < 4; ++r){
            for (int c = 0; c < 4; ++c){
                U_re[r][c] = U[r][c].real();
                U_im[r][c] = U[r][c].imag();
            }
        }
        const std::size_t s0 = std::size_t{1} << target_qubits[0];
        const std::size_t s1 = std::size_t{1} << target_qubits[1];
        for_each_quad(state, dim, s0, s1, thread_count, [=](complex128* __restrict__ s, const std::size_t i){
            const std::size_t idx[4] = {i, i + s0, i + s1, i + s0 + s1};
            double v_re[4], v_im[4];
            for (int c = 0; c < 4; ++c){
                v_re[c] = s[idx[c]].real();
                v_im[c] = s[idx[c]].imag();
            }
            for (int r = 0; r < 4; ++r){
                double sum_real = 0.0, sum_imag = 0.0;
                for (int c = 0; c < 4; ++c){
                    sum_real += U_re[r][c] * v_re[c] - U_im[r][c] * v_im[c];
                    sum_imag += U_re[r][c] * v_im[c] + U_im[r][c] * v_re[c];
                }
                s[idx[r]] = complex128(sum_real, sum_imag);
            }
        });
    }

    // Apply templated application of unitaries for 3 and 4 target qubits
    else if (num_targets == 3){
        apply_dense_unitary_fixed<3>(state, dim, U, target_qubits, thread_count);
    }
    else if (num_targets == 4){
        apply_dense_unitary_fixed<4>(state, dim, U, target_qubits, thread_count);
    }

    // Use the generic unitary application function for all other number of target qubits
    else {
        apply_dense_unitary_generic(state, dim, U, target_qubits, thread_count);
    }
}
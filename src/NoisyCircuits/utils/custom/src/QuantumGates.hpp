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
                p = end_pair;
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
    #pragma omp parallel num_threads(thread_count) if(quads >= MIN_PARALLEL_GOUPS)
    {
        const std::size_t num_threads = omp_get_num_threads();
        const std::size_t thread_number = omp_get_thread_num();
        std::size_t g = quads * thread_number / num_threads;
        const std::size_t end_g = quads * (thread_number + 1)  num_threads;
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
        s[j] = complex128(cosine * a - sine * d, cosine * d + sine * c);
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
    const std::size_t s11 = q1 + q2;
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
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
#include "QuantumGates.hpp"


// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Noise Application
// --------------------------------------------------------------------------------------------------------------------------------------


/**
 * Operator struct to hold the Kraus channels for single and two qubit gates
 */
struct KrausChannel{
    std::size_t num_targets = 0;        // Number of target qubits
    std::size_t num_ops = 0;            // Number of Kraus operators
    std::vector<complex128> kraus;      // K_k, operator blocks of size (d, d)
    std::vector<complex128> K_dag_K;    // Output of K_k^\dagger K_k for all operators
};

/**
 * Function to convert a set of kraus operators to a full channel
 * 
 * Inputs:
 *      noise_operators : const std::vector<matrix>&
 *          Reference to the set of Kraus operators into a KrausChannel
 * 
 * Returns:
 *      KrausChannel
 *          Kraus operators and their product of K^\dagger K
 */
static inline KrausChannel build_channel(const std::vector<matrix>& noise_operators){
    KrausChannel channel;
    channel.num_ops = noise_operators.size();
    if (noise_operators.empty()){
        return channel;
    }
    const std::size_t d = noise_operators[0].size();
    channel.num_targets = (d == 2) ? 1 : 2;
    channel.kraus.resize(channel.num_ops * d * d);
    channel.K_dag_K.resize(channel.num_ops * d * d);
    for (std::size_t k = 0; k < channel.num_ops; ++k){
        const matrix& K = noise_operators[k];
        complex128* kraus = channel.kraus.data() + k * d * d;
        complex128* kdagk = channel.K_dag_K.data() + k * d * d;
        for (std::size_t r = 0; r < d; ++r){
            for (std::size_t c = 0; c < d; ++c){
                kraus[r * d + c] = K[r][c];
                complex128 sum = 0.0;
                for (std::size_t m = 0; m < d; ++m){
                    sum += std::conj(K[m][r]) * K[m][r];
                }
                kdagk[r * d + c] = sum;
            }
        }
    }
    return channel;
}

/**
 * Function to compute the matrix-matrix product for two row-major matrices
 * 
 * Inputs:
 *      A : const complex128*
 *          Pointer to the first row-major matrix
 *      B : const complex128*
 *          Pointer to the second row-major matrix
 *      C : complex128*
 *          Pointer to the output row-major matrix
 *      d : const std::size_t
 *          Matrix dimension
 * 
 * Retturns:
 *      None
 */
static inline void matrix_mult(const complex128* __restrict__ A, const complex128* __restrict__ B, complex128* __restrict__ C, const std::size_t d){
    for (std::size_t r = 0; r < d; ++r){
        for (std::size_t c = 0; c < d; ++c){
            complex128 sum = 0.0;
            for (std::size_t m = 0; m < d; ++m){
                sum += A[r * d + m] * B[m * d + c];
            }
            C[r * d + c] = sum;
        }
    }
}

/**
 * Function that transforms the reduced density matrix with a gate: rho -> U rho U^\dagger
 * 
 * Inputs:
 *      rho : complex128*
 *          Pointer to the reduced density matrix
 *      U : const complex128*  
 *          Pointer to the gate unitary
 *      d : const std::size_t
 *          Dimension of the gate being applied to the state
 * 
 * Returns:
 *      None
 */
static inline void update_reduced_density_matrix(complex128* __restrict__ rho, const complex128* __restrict__ U, const std::size_t d){
    complex128 U_rho[16], U_dagger[16];
    matrix_mult(U, rho, U_rho, d);
    for (std::size_t r = 0; r < d; ++r){
        for (std::size_t c = 0; c < d; ++c){
            U_dagger[r * d + c] = std::conj(U[c * d + r]);
        }
    }
    matrix_mult(U_rho, U_dagger, rho, d);
}

/**
 * Function that computes the probabilities of all Kraus operators for a given qubit(s) and gate
 * 
 * Inputs:
 *      channel : const KrausChannel&
 *          Reference to the kraus channels of the noise instructions
 *      rho : const complex128*
 *          Reduced density matrix of the target qubit(s) in row-major format
 *      prob_list : std::vector<double>&
 *          Output vector reference to hold probabilities of the kraus ops
 * 
 * Returns:
 *      None
 */
static inline void get_kraus_probs(const KrausChannel& channel, const complex128* __restrict__ rho, std::vector<double>& prob_list){
    const std::size_t d = std::size_t{1} << channel.num_targets;
    prob_list.resize(channel.num_ops);
    for (std::size_t k = 0; k < channel.num_ops; ++k){
        const complex128* kdagk = channel.K_dag_K.data() + k * d * d;
        double probs = 0.0;
        for (std::size_t r = 0; r < d; ++r){
            for (std::size_t c = 0; c < d; ++c){
                probs += kdagk[r * d + c].real() * rho[c * d +r].real() - kdagk[r *d + c].imag() * rho[c * d + r].imag();
            }
        }
        prob_list[k] = std::max(probs, 0.0);
    }
}


// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Reduced Density Matrices
// --------------------------------------------------------------------------------------------------------------------------------------

/**
 * Function to compute the reduced density matrix from a statevector for a single qubit gate
 * 
 * Inputs:
 *      state : const complex128* 
 *          Pointer to the full statevector
 *      dim : const std::size_t
 *          Size of the statevector
 *      stride : const std::size_t
 *          2^q for qubit q
 *      rho : complex128*
 *          Output matrix to store the reduced density matrix
 * 
 * Returns:
 *      None
 */
static inline void get_reduced_density_matrix_single_qubit(const complex128* __restrict__ state, const std::size_t dim, const std::size_t stride, complex128* __restrict__ rho){
    double r00 = 0.0, r11 = 0.0, r01_real = 0.0, r01_imag = 0.0;
    for (std::size_t base = 0; base < dim; base += (stride << 1)){
        #pragma omp simd reduction(+:r00, r11, r01_real, r01_imag)
        for (std::size_t i = base; i < base + stride; ++i){
            const double a = state[i].real(), b = state[i].imag(), c = state[i + stride].real(), d = state[i + stride].imag();
            r00 += a * a + b * b;
            r11 += c * c + d * d;
            r01_real += a * c + b * d;
            r01_imag += b * c - a * d;
        }
    }
    rho[0] = complex128(r00, 0.0);
    rho[1] = complex128(r01_real, r01_imag);
    rho[2] = complex128(r01_real, -r01_imag);
    rho[3] = complex128(r11, 0.0);
}

/**
 * Function to compute the reduced density matrix from a statevector for a two qubit gate
 * 
 * Inputs:
 *      state : const complex128* 
 *          Pointer to the full statevector
 *      dim : const std::size_t
 *          Size of the statevector
 *      s1 : const std::size_t
 *          2^q1 for qubit q1
 *      s2 : const std::size_t
 *          2^q2 for qubit q2
 *      rho : complex128*
 *          Output matrix to store the reduced density matrix
 * 
 * Returns:
 *      None
 */
static inline void get_reduced_density_matrix_two_qubit(const complex128* __restrict__ state, const std::size_t dim, const std::size_t s1, const std::size_t s2, complex128* __restrict__ rho){
    const std::size_t l = std::min(s1, s2);
    const std::size_t h = std::max(s1, s2);
    const std::size_t s12 = s1 + s2;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    double r01_r = 0.0, r01_i = 0.0, r02_r = 0.0, r02_i = 0.0, r03_r = 0.0, r03_i = 0.0;
    double r12_r = 0.0, r12_i = 0.0, r13_r = 0.0, r13_i = 0.0, r23_r = 0.0, r23_i = 0.0;
    for (std::size_t out = 0; out < dim; out += (h << 1)){
        for (std::size_t mid = 0; mid < out + h; mid += (l << 1)){
            #pragma omp simd reduction(+:d0, d1, d2, d3, r01_r, r01_i, r02_r, r02_i, r03_r, r03_i, r12_r, r12_i, r13_r, r13_i, r23_r, r23_i)
            for (std::size_t i = mid; i < mid + l; ++i){
                const double x0 = state[i].real(), y0 = state[i].imag();
                const double x1 = state[i + s1].real(), y1 = state[i + s1].imag();
                const double x2 = state[i + s2].real(), y2 = state[i + s2].imag();
                const double x3 = state[i + s12].real(), y3 = state[i +s12].imag();
                
                // Diagonal terms
                d0 += x0 * x0 + y0 * y0;
                d1 += x1 * x1 + y1 * y1;
                d2 += x2 * x2 + y2 * y2;
                d3 += x3 * x3 + y3 * y3;

                // Off-diagonal terms
                r01_r += x0 * x1 + y0 * y1;
                r01_i += y0 * x1 - x0 * y1;
                r02_r += x0 * x2 + y0 * y2;
                r02_i += y0 * x2 - x0 * y2;
                r03_r += x0 * x3 + y0 * y3;
                r03_i += y0 * x3 - x0 * y3;
                r12_r += x1 * x2 + y1 * y2;
                r12_i += y1 * x2 - x1 * y2;
                r13_r += x1 * x3 + y1 * y3;
                r13_i += y1 * x3 - x1 * y3;
                r23_r += x2 * x3 + y2 * y3;
                r23_i += y2 * x3 - x2 * y3;
            }
        }
    }
    rho[0] = complex128(d0, 0.0);
    rho[1] = complex128(r01_r, r01_i);
    rho[2] = complex128(r02_r, r02_i);
    rho[3] = complex128(r03_r, r03_i);
    rho[4] = complex128(r01_r, -r01_i);
    rho[5] = complex128(d1, 0.0);
    rho[6] = complex128(r12_r, r12_i);
    rho[7] = complex128(r13_r, r13_i);
    rho[8] = complex128(r02_r, -r02_i);
    rho[9] = complex128(r12_r, -r12_i);
    rho[10] = complex128(d2, 0.0);
    rho[11] = complex128(r23_r, r23_i);
    rho[12] = complex128(r03_r, -r03_i);
    rho[13] = complex128(r13_r, -r13_i);
    rho[14] = complex128(r23_r, -r23_i);
    rho[15] = complex128(d3, 0.0);
}


// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for applying the local matrices to state
// --------------------------------------------------------------------------------------------------------------------------------------

/**
 * Function to apply a single qubit matrix to a state inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      A : const complex128*
 *          Pointer to the matrix
 *      dim : const std::size_t
 *          Length of the statevector
 *      stride : const std::size_t
 *          2^q for qubit q
 * 
 * Returns:
 *      None
 */
static inline void apply_matrix_single_qubit(complex128* __restrict__ state, const complex128* __restrict__ A, const std::size_t dim, const std::size_t stride){
    const double a00_r = A[0].real(), a00_i = A[0].imag(), a01_r = A[1].real(), a01_i = A[1].imag();
    const double a10_r = A[2].real(), a10_i = A[2].imag(), a11_r = A[3].real(), a11_i = A[3].imag();
    for_each_pair(state, dim, stride, 1, [=](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
        const double a = s[i].real(), b = s[i].imag(), c = s[j].real(), d = s[j].imag();
        s[i] = complex128(a00_r * a - a00_i * b + a01_r * c - a01_i * d, a00_r * b + a00_i * a + a01_r * d + a01_i * c);
        s[j] = complex128(a10_r * a - a10_i * b + a11_r * c - d * a11_i, a10_r * b + a10_i * a + a11_r * d + a11_i * c);
    });
}

/**
 * Function to apply a 2 qubit matrix to a statevector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      A : const complex128*
 *          Pointer to the matrix
 *      dim : const std::size_t
 *          Length of the statevector
 *      s1 : const std::size_t
 *          2^q1 for qubit q1
 *      s2 : const std::size_t
 *          2^q2 for qubit q2
 * 
 * Returns:
 *      None
 */
static inline void apply_matrix_two_qubit(complex128* __restrict__ state, const complex128* __restrict__ A, const std::size_t dim, const std::size_t s1, const std::size_t s2){
    double A_r[4][4], A_i[4][4];
    for (int r = 0; r < 4; ++r){
        for (int c = 0; c < 4; ++c){
            A_r[r][c] = A[r * 4 + c].real();
            A_i[r][c] = A[r * 4 + c].imag();
        }
    }
    for_each_quad(state, dim, s1, s2, 1, [=](complex128* __restrict__ s, const std::size_t i){
        const std::size_t idx[4] = {i, i + s1, i + s2, i + s1 + s2};
        double v_r[4], v_i[4];
        for (int c = 0; c < 4; ++c){
            v_r[c] = s[idx[c]].real();
            v_i[c] = s[idx[c]].imag();
        }
        for (int r = 0; r < 4; ++r){
            double sum_r = 0.0, sum_i = 0.0;
            for (int c = 0; c < 4; ++c){
                sum_r += A_r[r][c] * v_r[c] - A_i[r][c] * v_i[c];
                sum_i += A_r[r][c] * v_i[c] + A_i[r][c] * v_r[c];
            }
            s[idx[r]] = complex128(sum_r, sum_i);
        }
    });
}

// --------------------------------------------------------------------------------------------------------------------------------------
// Code Section for Noise Application
// --------------------------------------------------------------------------------------------------------------------------------------

/**
 * Function to compute the reduced density matrix for the target qubit(s)
 * 
 * Inputs:
 *      state : const complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Index of qubit 1
 *      q2 : const std::size_t
 *          Index of qubit 2
 *      num_qubits : const std::size_t
 *          Total number of qubits
 *      num_targets : const std::size_t
 *          Total number of target qubits for noise and gate
 *      rho : complex128*
 *          Pointer to store the reduced density matrix
 * 
 * Returns:
 *      None
 */
static inline void get_reduced_density_matrix(const complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const std::size_t num_targets, complex128* __restrict__ rho){
    const std::size_t dim = std::size_t{1} << num_qubits;
    if (num_targets == 1){
        get_reduced_density_matrix_single_qubit(state, dim, std::size_t{1} << q1, rho);
    }
    else {
        get_reduced_density_matrix_two_qubit(state, dim, std::size_t{1} << q1, std::size_t{1} << q2, rho);
    }
}

/**
 * Function that applies the noise to the statevector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t 
 *          Target qubit index 1
 *      q2 : const std::size_t
 *          Target qubit index 2 (unused for single qubit gates)
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      channel : const KrausChannel&
 *          Kraus channel of the instruction
 *      rho : const complex128*
 *          Pointer to the reduced density matrix
 *      gate : const complex128*
 *          Pointer to gate matrix U
 *      traj_engine : std::mt19937_64&
 *          Reference to the RNG engine
 * 
 * Returns:
 *      None
 */
static inline void apply_noise_channel(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const KrausChannel& channel, const complex128* __restrict__ rho, const complex128* __restrict__ gate, std::mt19937_64& traj_engine){
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t d = std::size_t{1} << channel.num_targets;
    std::vector<double> probs_list;
    get_kraus_probs(channel, rho, probs_list);
    std::discrete_distribution<> dist(probs_list.begin(), probs_list.end());
    const std::size_t c = dist(traj_engine);
    const complex128* kraus = channel.kraus.data() + c * d * d;
    complex128 A[16];
    if (gate != nullptr){
        matrix_mult(kraus, gate, A, d);
    }
    else {
        std::copy(kraus, kraus + d * d, A);
    }
    const double one_by_sqrt_p = 1.0 / std::sqrt(probs_list[c]);
    for (std::size_t e = 0; e < d * d; ++e){
        A[e] *= one_by_sqrt_p;
    }
    if (channel.num_targets == 1){
        apply_matrix_single_qubit(state, A, dim, std::size_t{1} << q1);
    }
    else {
        apply_matrix_two_qubit(state, A, dim, std::size_t{1} << q1, std::size_t{1} << q2);
    }
}

/**
 * Function to apply the noisy gate to the statevector inplace
 * 
 * Inputs:
 *      state : complex128*
 *          Pointer to the statevector
 *      q1 : const std::size_t
 *          Target qubit index 1
 *      q2 : const std::size_t
 *          Target qubit index 2 (unused for single qubit gates)
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      gate : const complex128*
 *          Pointer to the gate unitary
 *      channel : const KrausChannel&
 *          Reference to the kraus channel for the instruction
 *      traj_engine : std::mt19937_64&
 *          Reference to the RNG engine
 * 
 * Returns:
 *      None
 */
static inline void apply_noise_gate(complex128* __restrict__ state, const std::size_t q1, const std::size_t q2, const std::size_t num_qubits, const complex128* __restrict__ gate, const KrausChannel& channel, std::mt19937_64& traj_engine){
    const std::size_t d = std::size_t{1} << channel.num_targets;
    complex128 rho[16];
    get_reduced_density_matrix(state, q1, q2, num_qubits, channel.num_targets, rho);
    update_reduced_density_matrix(rho, gate, d);
    apply_noise_channel(state, q1, q2, num_qubits, channel, rho, gate, traj_engine);
}
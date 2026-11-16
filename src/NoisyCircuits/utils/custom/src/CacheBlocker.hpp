/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * 
 */

#pragma once
#include "TypeDefs.hpp"
#include "QuantumGates.hpp"

using GateFunction = void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint);

// Number of qubits that perfectly into the L2 cache of a core --> Set to 1MiB which corresponds to 16 qubits
constexpr std::size_t CACHE_BLOCK_QUBITS = 16;
// Smaller statevectors can stay in the L2/L3 cache without blocking required
constexpr std::size_t MIN_CACHE_BLOCKING_QUBITS = 20;
// Minimum number of chunks per thread
constexpr std::size_t MIN_CHUNKS_PER_THREAD = 4;
// Lower bound on the chunk size when shrinking for many threads
constexpr std::size_t MIN_BLOCK_QUBITS = 10;


/**
 * Defines the application of a gate to 1 chunk
 */
enum class ChunkApplication : unsigned char {
    Global,         // Acts non-diagonally on a high qubit
    Skip,           // Identity gate added
    Kernel,         // All qubits applied are low
    Phase,          // All qubits applied are high
    ZOnLow,         // CZ gate with a high control and low target
    XOnLow,         // CX gate with a high control and low target
    RZOnLow,        // RZZ gate with one high qubit and one low qubit
    DiagonalOnLow   // Diagonal unitary on high and low qubits
};

/**
 * Struct to hold per computed data
 */
struct BlockedGate{
    ChunkApplication application = ChunkApplication::Global;
    std::size_t q1 = 0;
    std::size_t q2 = 0;
    std::vector<std::size_t> high_bits;
    std::vector<complex128> phases;
    std::size_t low_qubit = 0;
    std::vector<std::size_t> low_targets;
    std::vector<matrix> reduced_diagonals;
};

/**
 * Struct to hold a single stage --> either chunk-local gates applied by the chunk or the single gate applied to the full state
 */
struct BlockedStage{
    bool local;
    std::vector<std::size_t> gates;
};

/**
 * Function that returns the second qubit argument passed to the gate and the noise functions
 * 
 * Inputs:
 *      qubits : const std::vector<std::size_t>&
 *          Reference to the qubits of the given instruction
 * 
 * Returns:
 *      std::size_t 
 *          qubits[1] if the instruction has more than 1 qubit, otherwise qubit 0
 */
static inline std::size_t second_qubit(const std::vector<std::size_t>& qubits){
    return qubits.size() > 1 ? qubits[1] : qubits[0];
}

/**
 * Function to collect the values of high qubits of a chunk into a single index
 * 
 * Inputs:
 *      chunk : const std::size_t
 *          Index of the chunk
 *      high_bits : const std::vector<std::size_t>&
 *          Reference to the bit positions of the high qubits
 * 
 * Return:
 *      std::size_t
 *          Index whose bit j is the value of the high qubit
 */
static inline std::size_t high_qubit_values(const std::size_t chunk, const std::vector<std::size_t>& high_bits){
    std::size_t values = 0;
    for (std::size_t j = 0; j < high_bits.size(); ++j){
        values |= ((chunk >> high_bits[j]) & 1) << j;
    }
    return values;
}

/**
 * Function to multiply all amplitudes of a chunk by a single phase
 * 
 * Inputs:
 *      chunk : complex128*
 *          Pointer to the first amplitude of the chunk
 *      chunk_dim : const std::size_t
 *          Number of amplitudes in the chunk
 *      phase : const complex128
 *          Phase factor
 * 
 * Returns:
 *      None
 */
static inline void scale_chunk(complex128* __restrict__ chunk, const std::size_t chunk_dim, const complex128 phase){
    const double phase_real = phase.real(), phase_imag = phase.imag();
    #pragma omp simd
    for (std::size_t i = 0; i < chunk_dim; ++i){
        const double a = chunk[i].real(), b = chunk[i].imag();
        chunk[i] = complex128(phase_real * a - phase_imag * b, phase_real * b + phase_imag * a);
    }
}

/**
 * Function to classify a given unitary gate
 * 
 * Inputs:
 *      U : const matrix&
 *          Reference to the unitary matrix
 *      diagonal : bool& 
 *          Set to true if the unitary operator has no off-diagonal elements
 *      identity : bool&
 *          Set to true if the unitary operator is an Identity matrix
 * 
 * Returns:
 *      None
 */
static inline void classify_unitary_matrix(const matrix& U, bool& diagonal, bool& identity){
    diagonal = true;
    identity = true;
    for (std::size_t r = 0; r < U.size(); ++r){
        for (std::size_t c = 0; c < U.size(); ++c){
            if (r != c && U[r][c] != 0.0){
                diagonal = false;
                identity = false;
                return;
            }
            if (r == c && U[r][c] != 1.0){
                identity = false;
            }
        }
    }
}

/**
 * Function to classify how a gate must be applied to a chunk and also precompute the required data
 * 
 * Inputs:
 *      instruction : const ItemEntry&
 *          Reference to the instruction
 *      block_qubits : const std::size_t       
 *          Number of low qubits b (a chunk holds 2^b amplitudes)
 * 
 * Returns:
 *      BlockedGate
 *          Application and the precomputed data
 */
static BlockedGate classify_gate(const ItemEntry& instruction, const std::size_t block_qubits){
    BlockedGate gate;
    const std::vector<std::size_t>& qubits = instruction.qubits;
    const std::string& name = instruction.gate_name;
    const double theta = instruction.params;
    gate.q1 = qubits[0];
    gate.q2 = second_qubit(qubits);
    bool diagonal = false, identity = false;

    if (name == "unitary"){
        classify_unitary_matrix(instruction.unitary_matrix, diagonal, identity);
        if (identity){
            gate.application = ChunkApplication::Skip;
            return gate;
        }
    }
    bool all_low = true;
    for (const std::size_t q : qubits){
        all_low = all_low && q < block_qubits;
    }
    if (all_low){
        gate.application = ChunkApplication::Kernel;
        return gate;
    }

    const bool q1_h = gate.q1 >= block_qubits, q2_h = gate.q2 >= block_qubits;
    if (name == "rz"){
        gate.application = ChunkApplication::Phase;
        gate.high_bits = {gate.q1 - block_qubits};
        gate.phases = {std::polar(1.0, -0.5 * theta), std::polar(1.0, 0.5 * theta)};
    }
    else if (name == "p"){
        gate.application = ChunkApplication::Phase;
        gate.high_bits = {gate.q1 - block_qubits};
        gate.phases = {complex128(1.0, 0.0), std::polar(1.0, theta)};
    }
    else if (name == "cz" && q1_h && q2_h){
        gate.application = ChunkApplication::Phase;
        gate.high_bits = {gate.q1 - block_qubits, gate.q2 - block_qubits};
        gate.phases = {complex128(1.0, 0.0), complex128(1.0, 0.0), complex128(1.0, 0.0), complex128(-1.0, 0.0)};
    }
    else if (name == "cz"){
        gate.application = ChunkApplication::ZOnLow;
        gate.high_bits = {(q1_h ? gate.q1 : gate.q2) - block_qubits};
        gate.low_qubit = q1_h ? gate.q2 : gate.q1;
    }
    else if (name == "rzz" && q1_h && q2_h){
        const complex128 same = std::polar(1.0, -0.5 * theta), different = std::polar(1.0, 0.5 * theta);
        gate.application = ChunkApplication::Phase;
        gate.high_bits = {gate.q1 - block_qubits, gate.q2 - block_qubits};
        gate.phases = {same, different, different, same};
    }
    else if (name == "rzz"){
        gate.application = ChunkApplication::RZOnLow;
        gate.high_bits = {(q1_h ? gate.q1 : gate.q2) - block_qubits};
        gate.low_qubit = q1_h ? gate.q2 : gate.q1;
    }
    else if (name == "cx" && q1_h && !q2_h){
        gate.application = ChunkApplication::XOnLow;
        gate.high_bits = {gate.q1 - block_qubits};
        gate.low_qubit = gate.q2;
    }
    else if (name == "unitary" && diagonal){
        std::vector<std::size_t> high_position, low_position;
        for (std::size_t j = 0; j < qubits.size(); ++j){
            if (qubits[j] >= block_qubits){
                gate.high_bits.push_back(qubits[j] - block_qubits);
                high_position.push_back(j);
            }
            else{
                gate.low_targets.push_back(qubits[j]);
                low_position.push_back(j);
            }
        }
        const std::size_t num_high_vals = std::size_t{1} << high_position.size();
        const std::size_t low_dim = std::size_t{1} << low_position.size();
        const matrix& U = instruction.unitary_matrix;
        for (std::size_t h = 0; h < num_high_vals; ++h){
            std::size_t high_row = 0;
            for (std::size_t j = 0; j < high_position.size(); ++j){
                high_row |= ((h >> j) & 1) << high_position[j];
            }
            if (low_position.empty()){
                gate.phases.push_back(U[high_row][high_row]);
                continue;
            }
            matrix reduced(low_dim, std::vector<complex128>(low_dim, 0.0));
            for (std::size_t m = 0; m < low_dim; ++m){
                std::size_t row = high_row;
                for (std::size_t j = 0; j < low_position.size(); ++j){
                    row |= ((m >> j) & 1) << low_position[j];
                }
                reduced[m][m] = U[row][row];
            }
            gate.reduced_diagonals.push_back(std::move(reduced));
        }
        gate.application = low_position.empty() ? ChunkApplication::Phase : ChunkApplication::DiagonalOnLow;
    }
    else {
        gate.application = ChunkApplication::Global;
    }
    return gate;
}

/**
 * Function to group gates into stages --> Every gate gets 1 round where even rounds are local stages and odd rounds are global gates. A gate goes into the earliest round of its kind that is not earlier than the round of the last gate on any of its qubits, so gates that share a qubit keep their order. Within a round the gates keep their order.
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the instructions of the circuit
 *      blocked_gates : cosnt std::vector<BlockedGate>&
 *          Reference to the chunk application of the instruction
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 * 
 * Returns:
 *      std::vector<BlockedStage>
 *          Stages in the order in which they are applied
 */
static std::vector<BlockedStage> build_stages(const std::vector<ItemEntry>& instruction_list, const std::vector<BlockedGate>& blocked_gates, const std::size_t num_qubits){
    std::vector<std::size_t> last_round(num_qubits, 0);
    std::vector<std::vector<std::size_t>> rounds;
    for (std::size_t g = 0; g < instruction_list.size(); ++g){
        const ChunkApplication application = blocked_gates[g].application;
        if (application == ChunkApplication::Skip){
            continue;
        }
        std::size_t round = 0;
        for (const std::size_t q : instruction_list[g].qubits){
            round = std::max(round, last_round[q]);
        }
        const bool local = application != ChunkApplication::Global;
        if (local != (round % 2 == 0)){
            ++round;
        }
        for (const std::size_t q : instruction_list[g].qubits){
            last_round[q] = round;
        }
        if (rounds.size() <= round){
            rounds.resize(round + 1);
        }
        rounds[round].push_back(g);
    }
    std::vector<BlockedStage> stages;
    for (std::size_t round = 0; round < rounds.size(); ++round){
        if (rounds[round].empty()){
            continue;
        }
        if (round % 2 == 0){
            stages.push_back(BlockedStage{true, std::move(rounds[round])});
        }
        else{
            for (const std::size_t g : rounds[round]){
                stages.push_back(BlockedStage{false, {g}});
            }
        }
    }
    return stages;
}

/**
 * Function that applies a chunk-local gate to one chunk
 * 
 * Inputs:
 *      chunk : complex128*    
 *          Pointer to the first amplitude of a chunk
 *      chunk_index : const std::size_t
 *          Index of the chunk
 *      block_qubits : const std::size_t
 *          Number of low qubits
 *      instruction : const ItemEntry&
 *          Reference to the instruction
 *      gate : const BlockedGate&
 *          Reference to the chunk application of the instruction
 *      gate_function : const GateFunction
 *          Gate function of the instruction
 * 
 * Returns:
 *      None
 */
static inline void apply_gate_to_chunk(complex128* __restrict__ chunk, const std::size_t chunk_index, const std::size_t block_qubits, const ItemEntry& instruction, const BlockedGate& gate, const GateFunction gate_function){
    const std::size_t chunk_dim = std::size_t{1} << block_qubits;
    switch (gate.application){
        case ChunkApplication::Kernel:
            gate_function(chunk, gate.q1, gate.q2, block_qubits, instruction.params, instruction.unitary_matrix, instruction.qubits, 1);
            break;
        case ChunkApplication::Phase: {
            const complex128 phase = gate.phases[high_qubit_values(chunk_index, gate.high_bits)];
            if (phase != complex128(1.0, 0.0)){
                scale_chunk(chunk, chunk_dim, phase);
            }
            break;
        }
        case ChunkApplication::ZOnLow:
            if (high_qubit_values(chunk_index, gate.high_bits)){
                for_each_pair(chunk, chunk_dim, std::size_t{1} << gate.low_qubit, 1, [](complex128* __restrict__ s, const std::size_t i, const std::size_t j){
                    s[j] = -s[j];
                });
            }
            break;
        case ChunkApplication::XOnLow:
            if (high_qubit_values(chunk_index, gate.high_bits)){
                apply_X_gate(chunk, gate.low_qubit, gate.low_qubit, block_qubits, 0.0, instruction.unitary_matrix, instruction.qubits, 1);
            }
            break;
        case ChunkApplication::RZOnLow: {
            const double theta = high_qubit_values(chunk_index, gate.high_bits) ? -instruction.params : instruction.params;
            apply_RZ_gate(chunk, gate.low_qubit, gate.low_qubit, block_qubits, theta, instruction.unitary_matrix, instruction.qubits, 1);
            break;
        }
        case ChunkApplication::DiagonalOnLow:
            apply_diagonal_matrix(chunk, chunk_dim, gate.reduced_diagonals[high_qubit_values(chunk_index, gate.high_bits)], gate.low_targets, 1);
            break;
        default:
            break;
    }
}

/**
 * Function to choose the number of low qubits b or returns 0 if the state is too small for cache blocking
 * 
 * Inputs:
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      thread_count : const unsigned int
 *          Number of threads
 * 
 * Returns:
 *      std::size_t
 *          Number of low qubits b, or 0 if the statevector is too small
 */
static inline std::size_t choose_block_qubits(const std::size_t num_qubits, cuint thread_count){
    if (num_qubits < MIN_CACHE_BLOCKING_QUBITS){
        return 0;
    }
    std::size_t block_qubits = std::min(CACHE_BLOCK_QUBITS, num_qubits);
    const std::size_t min_chunks = MIN_CHUNKS_PER_THREAD * std::max<std::size_t>(thread_count, 1);
    while (block_qubits > MIN_BLOCK_QUBITS && (std::size_t{1} << (num_qubits - block_qubits)) < min_chunks){
        --block_qubits;
    }
    return block_qubits;
}

/**
 * Function that applies all instructions to the statevector with cache blocking
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the instructions of the circuit
 *      gate_functions : const std::vector<GateFunction>&
 *          Gate function of each instruction
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      state : complex128*
 *          Pointer to the statevector
 *      block_qubits : const std::size_t
 *          Number of low qubits (b), 1 <= b <= num_qubits
 *      thread_count : const unsigned int
 *          Number of threads
 * 
 * Returns:
 *      None
 */
static inline void apply_gates_blocked(const std::vector<ItemEntry>& instruction_list, const std::vector<GateFunction>& gate_functions, const std::size_t num_qubits, complex128* __restrict__ state, const std::size_t block_qubits, cuint thread_count){
    std::vector<BlockedGate> blocked_gates;
    blocked_gates.reserve(instruction_list.size());
    for (const ItemEntry& instruction : instruction_list){
        blocked_gates.push_back(classify_gate(instruction, block_qubits));
    }
    const std::vector<BlockedStage> stages = build_stages(instruction_list, blocked_gates, num_qubits);
    const std::size_t num_chunks = std::size_t{1} << (num_qubits - block_qubits);
    for (const BlockedStage& stage : stages){
        if (stage.local){
            #pragma omp parallel for schedule(static) num_threads(thread_count)
            for (std::size_t c = 0; c < num_chunks; ++c){
                complex128* chunk = state + (c << block_qubits);
                for (const std::size_t g : stage.gates){
                    apply_gate_to_chunk(chunk, c, block_qubits, instruction_list[g], blocked_gates[g], gate_functions[g]);
                }
            }
        }
        else {
            const std::size_t g = stage.gates[0];
            const BlockedGate& gate = blocked_gates[g];
            gate_functions[g](state, gate.q1, gate.q2, num_qubits, instruction_list[g].params, instruction_list[g].unitary_matrix, instruction_list[g].qubits, thread_count);
        }
    }
}
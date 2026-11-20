/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This source code provides the functionality required to execute quantum circuit simulations from python through the NoisyCircuits library within a shared memory environment.
 */

#include "TypeDefs.hpp"
#include "NoiseParser.hpp"
#include "QuantumGates.hpp"
#include "NoiseApplication.hpp"
#include "FunctionMapper.hpp"
#include "CacheBlocker.hpp"
#include "GroupTrajectories.hpp"


/**
 * Struct to store noise instructions for an instruction
 */
struct NoiseInstruction{
    const KrausChannel* channel = nullptr;
    std::array<complex128, 16> gate{}; 
};

/**
 * Function to look up the gate function of each instruction once, before execution
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the list of instructions that are required to build the quantum circuit.
 * 
 * Returns:
 *      std::vector<GateFunction>
 *          Function that applies the gate for instruction g at position p
 */
static std::vector<GateFunction> resolve_gate_functions(const std::vector<ItemEntry>& instruction_list){
    const auto gate_map = gate_function_mapper();
    std::vector<GateFunction> gate_functions;
    gate_functions.reserve(instruction_list.size());
    for (const ItemEntry& instruction : instruction_list){
        const auto gate = gate_map.find(instruction.gate_name);
        if (gate == gate_map.end()){
            throw std::invalid_argument("Unsupported Gate: " + instruction.gate_name);
        }
        gate_functions.push_back(gate->second);
    }
    return gate_functions;
}

/**
 * Function that inplace evolves the state of a quantum circuit using the pure statevector simulation
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the vector constaining the quantum circuit instructions
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit
 *      state : complex128*
 *          Pointer to the statevector
 *      return_state : bool
 *          A flag to specify whether to return the statevector or the probabilities associated to the statevector.
 *      thread_count : const unsigned int
 *          Total number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
void pure_state_solver(const std::vector<ItemEntry>& instruction_list, const std::size_t num_qubits, complex128* __restrict__ state, bool return_state, cuint thread_count){
    const std::vector<GateFunction> gate_functions = resolve_gate_functions(instruction_list);
    state[0] = complex128(1.0, 0.0);
    const std::size_t block_qubits = choose_block_qubits(num_qubits, thread_count);
    if (block_qubits > 0){
        apply_gates_blocked(instruction_list, gate_functions, num_qubits, state, block_qubits, thread_count);
    }
    else{
        for (std::size_t g = 0; g < instruction_list.size(); ++g){
            const ItemEntry& instruction = instruction_list[g];
            const std::vector<std::size_t>& qubits = instruction.qubits;
            gate_functions[g](state, qubits[0], second_qubit(qubits), num_qubits, instruction.params, instruction.unitary_matrix, qubits, thread_count);
        }
    }
    if (!return_state){
        const std::size_t dim = std::size_t{1} << num_qubits;
        #pragma omp parallel for num_threads(thread_count) if(dim >= (MIN_PARALLEL_GROUPS << 1))
        for (std::size_t i = 0; i < dim; i++){
            const complex128 amplitude = state[i];
            state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
        }
    }
}

/**
 * Functio that computes the matrix for a gate on the target qubit(s) by applying the gate function to basis states
 * 
 * Inputs:
 *      gate_function : const GateFunction
 *          Function to apply the gate
 *      instruction : const ItemEntry&
 *          Reference to the instruction
 *      num_targets : const std::size_t
 *          Number of target qubits
 *      U : complex128*
 *          Pointer to the gate unitary
 * 
 * Returns:
 *      None
 */
static void get_gate_matrix(const GateFunction gate_function, const ItemEntry& instruction, const std::size_t num_targets, complex128* __restrict__ U){
    const std::size_t dim = std::size_t{1} << num_targets;
    complex128 column_vector[4];
    for (std::size_t c = 0; c < dim; ++c){
        std::fill(column_vector, column_vector + dim, complex128(0.0, 0.0));
        column_vector[c] = complex128(1.0, 0.0);
        gate_function(column_vector, 0, num_targets - 1, num_targets, instruction.params, instruction.unitary_matrix, instruction.qubits, 1);
        for (std::size_t r = 0; r < dim; ++r){
            U[r * dim + c] = column_vector[r];
        }
    }
}

/**
 * Function that computes the fully evolved statevector for a single trajectory of the Monte-Carlo Wavefunction method
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the instructions to build and execute the quantum circuit
 *      gate_functions : const std::vector<GateFunction>&
 *          Function that applies the gate of each instruction
 *      noise_instructions : const std::vector<NoiseInstruction>&
 *          Reference to the kraus channel and gate matrix
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      seed : const unsigned int
 *          Unique seed value for th RNG engine
 * 
 * Returns:
 *      std::vector<complex128>
 *          Probabilities of the statevector from the trajectory evolution
 */
std::vector<complex128> single_trajectory(const std::vector<ItemEntry>& instruction_list, const std::vector<GateFunction>& gate_functions, const std::vector<NoiseInstruction>& noise_instructions, const std::size_t num_qubits, cuint seed){
    std::mt19937_64 trajectory_engine(seed);
    std::vector<complex128> trajectory_state(std::size_t{1} << num_qubits, 0.0);
    trajectory_state[0] = complex128(1.0, 0.0);
    for (std::size_t g = 0; g < instruction_list.size(); ++g){
        const ItemEntry& instruction = instruction_list[g];
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const std::size_t q2 = second_qubit(qubits);
        const KrausChannel& channel = *noise_instructions[g].channel;
        if (channel.num_targets > 0){
            apply_noise_gate(trajectory_state.data(), qubits[0], q2, num_qubits, noise_instructions[g].gate.data(), channel, trajectory_engine);
        }
        else {
            gate_functions[g](trajectory_state.data(), qubits[0], q2, num_qubits, instruction.params, instruction.unitary_matrix, qubits, 1);
        }
    }
    for (std::size_t i = 0; i < trajectory_state.size(); ++i){
        const complex128 amp = trajectory_state[i];
        trajectory_state[i] = complex128(amp.real() * amp.real() + amp.imag() * amp.imag());
    }
    return trajectory_state;
}

/**
 * Function that performs the Monte-Carlo Wavefunction to approximate the density matrix simulation of a given quantum circuit with a noise instruction
 * 
 * Inputs:
 *      instruction_list : const std::vector<ItemEntry>&
 *          Reference to the list of instructions required to build and execute the quantum circuit
 *      state : complex128*
 *          Pointer to the statevector
 *      single_qubit_instructions : const std::vector<noise_map>&
 *          Reference to the set of single qubit noise instructions
 *      two_qubit_instructions : const noise_map2q&
 *          Reference to the set of two qubit noise instructions
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      num_trajectories : const unsigned int
 *          Total number of trajectories to simulate
 *      thread_count : const unsigned int
 *          Total number of threads to distribute computation
 * 
 * Returns:
 *      None
 */
void monte_carlo_wavefunction_solver(const std::vector<ItemEntry>& instruction_list, complex128* __restrict__ state, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, cuint num_trajectories, cuint thread_count){
    const std::vector<GateFunction> gate_functions = resolve_gate_functions(instruction_list);
    std::unordered_map<const std::vector<matrix>*, std::size_t> channel_index;
    std::vector<std::size_t> instruction_channel(instruction_list.size(), 0);
    std::vector<KrausChannel> channels(1);
    for (std::size_t g = 0; g < instruction_list.size(); ++g){
        const ItemEntry& instruction = instruction_list[g];
        if (instruction.gate_name == "unitary"){
            continue;
        }
        const std::vector<matrix>* operators = &get_matrix_list_for_instruction(instruction.gate_name, single_qubit_instructions, two_qubit_instructions, instruction.qubits[0], second_qubit(instruction.qubits));
        const auto found_index_in_channel = channel_index.find(operators);
        if (found_index_in_channel != channel_index.end()){
            instruction_channel[g] = found_index_in_channel -> second;
        }
        else {
            channel_index.emplace(operators, channels.size());
            instruction_channel[g] = channels.size();
            channels.push_back(build_channel(*operators));
        }
    }
    std::vector<NoiseInstruction> noise_instructions(instruction_list.size());
    for (std::size_t g = 0; g < instruction_list.size(); ++g){
        noise_instructions[g].channel = &channels[instruction_channel[g]];
        if (noise_instructions[g].channel -> num_targets > 0){
            get_gate_matrix(gate_functions[g], instruction_list[g], noise_instructions[g].channel->num_targets, noise_instructions[g].gate.data());
        }
    }
    // double averaging_factor = 1.0 / static_cast<double>(num_trajectories);
    // #pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    // for (unsigned int t = 0; t < num_trajectories; ++t){
    //     cuint seed = static_cast<cuint>(42 + t);
    //     std::vector<complex128> trajectory_state = single_trajectory(instruction_list, gate_functions, noise_instructions, num_qubits, seed);
    //     #pragma omp critical
    //     {
    //         for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
    //             state[i] += trajectory_state[i] * averaging_factor;
    //         }
    //     }
    // }
    const unsigned group_size  = auto_group_size(num_qubits, thread_count);
    monte_carlo_wavefunction_grouped(instruction_list, gate_functions, noise_instructions, state, num_qubits, num_trajectories, thread_count, group_size);
}

/**
 * Main method to act as interface to simulate a given quantum circuit with/without noise
 * 
 * Inputs:
 *      instructions : py::list
 *          List of instructions to build the quantum circuit obtained from python
 *      statevector : py::array_t<complex128>
 *          A numpy array of dtype np.complex128 for inplace modification of the statevector
 *      single_qubit_noise_instructions : py::dict
 *          Dictionary of single qubit noise instructions obtained from python
 *      two_qubit_noise_instructions : py::dict
 *          Dictionary of two qubit noise instructions obtained from python
 *      num_qubits : std::size_t
 *          Total number of qubits in the system
 *      noisy : bool
 *          Flag that is used to determine whether to perform a noise-free or noise-aware simulation
 *      num_trajectories : const unsigned int
 *          Total number of trajectories to simulate when performing the noise-aware simulations
 *      return_state : bool
 *          Flag used for pure statevector (noise-free) simulations to determine whether to return the state or it's probabilities
 *      thread_count : const unsigned int
 *          Total number of threads to distribute computation
 * 
 * Returns:
 *      None
 */
void simulate_circuit(py::list instructions, py::array_t<complex128> statevector, py::dict single_qubit_noise_instructions, py::dict two_qubit_noise_instructions, std::size_t num_qubits, bool noisy, cuint num_trajectories, bool return_state, cuint thread_count){
    std::vector<ItemEntry> instruction_list;
    instruction_list.reserve(py::len(instructions));
    for (auto item : instructions){
        auto item_tuple = item.cast<py::tuple>();
        ItemEntry entry;
        entry.gate_name = item_tuple[0].cast<std::string>();
        std::vector<std::size_t> apply_to_qubits = item_tuple[1].cast<std::vector<std::size_t>>();
        if (entry.gate_name == "unitary"){
            py::array unitary_matrix_array = py::cast<py::array>(item_tuple[2]);
            auto arr = py::array_t<complex128, py::array::c_style | py::array::forcecast>(unitary_matrix_array);
            auto a = arr.unchecked<-1>();
            matrix U(a.shape(0), std::vector<complex128>(a.shape(1)));
            std::size_t dim = a.shape(0);
            for (std::size_t i = 0; i < dim; i++){
                #pragma GCC unroll 2
                for (std::size_t j = 0; j < dim; j++){
                    U[i][j] = a(i, j);
                }
            }
            entry.unitary_matrix = std::move(U);
            entry.params = -1.0;
            std::reverse(apply_to_qubits.begin(), apply_to_qubits.end());
            entry.qubits = std::move(apply_to_qubits);
        }
        else {
            entry.unitary_matrix = {};
            entry.qubits = std::move(apply_to_qubits);
            entry.params = item_tuple[2].cast<double>();
        }
        instruction_list.push_back(std::move(entry));
    }
    py::buffer_info state_info = statevector.request();
    auto * state = static_cast<complex128*>(state_info.ptr);
    if (noisy){
        std::vector<noise_map> single_qubit_instructions = parse_single_qubit_noise(single_qubit_noise_instructions);
        noise_map2q two_qubit_instructions = parse_two_qubit_noise(two_qubit_noise_instructions);
        monte_carlo_wavefunction_solver(instruction_list, state, single_qubit_instructions, two_qubit_instructions, num_qubits, num_trajectories, thread_count);
    }
    else {
        unsigned short max_usable_threads = std::min<unsigned short>(thread_count, static_cast<unsigned short>(omp_get_max_threads()));
        pure_state_solver(instruction_list, num_qubits, state, return_state, max_usable_threads);
    }
}

/**
 * Binder for syncing C++ code as a shared library to python.
 */
PYBIND11_MODULE(simulator, m){
    m.doc() = "Module for simulating quantum circuits with noise using the Monte Carlo Wavefunction method.";
    m.def("simulate_circuit", &simulate_circuit, "Simulates a quantum circuit with and without noise.", py::arg("instructions"), py::arg("statevector"), py::arg("single_qubit_noise_instructions"), py::arg("two_qubit_noise_instructions"), py::arg("num_qubits"), py::arg("noisy"), py::arg("num_trajectories"), py::arg("return_state"), py::arg("thread_count"));
}
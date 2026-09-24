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

/**
 * Function that inplace evolves the state of a quantum circuit using the pure statevector simulation (no noise)
 * 
 * Inputs:
 *      instruction_list : const std::list<ItemEntry>&
 *          Reference to the list of instructions that are required to build the quantum circuit. Each entry in the list is a struct of ItemEntry
 *      num_qubits : std::size_t
 *          Total number of qubits in the system
 *      state : complex128*
 *          Pointer to the statevector
 *      return_sate : bool
 *          A flag to specify whether to return the statevector or the probabilities associated to the statevector.
 *      thread_count : const unsigned int
 *          Total number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
void pure_state_solver(const std::list<ItemEntry>& instruction_list, std::size_t num_qubits, complex128* __restrict__ state, bool return_state, cuint thread_count){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)> gate_map = gate_function_mapper();
    state[0] = complex128(1.0, 0.0);
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](state, qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, thread_count);
    }
    if (!return_state) {
        #pragma omp parallel for num_threads(thread_count)
        for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
            const complex128 amplitude = state[i];
            state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
        }
    }
}

/**
 * Function that computes the fully evolved statevector for a single trajectory of the Monte-Carlo Wavefunction method.
 * 
 * Inputs:
 *      instruction_list : const std::list<ItemEntry>&
 *          Reference to the list of instructions that are required to build the quantum circuit. Each entry in the list is a struct of ItemEntry
 *      single_qubit_instructions : const std::vector<noise_map>& 
 *          Reference to entire set of single qubit noise instructions (for all gates and qubits)
 *      two_qubit_instructions : const noise_map2q&
 *          Reference to entire set of two qubit noise instructions (for all gates and qubits)
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      gate_map : std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, const unsigned int)>
 *          Unordered map that has a key - value pair of gate name (string) and the function that applies the gate          
 *      noise_map : std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, const unsigned int)>
 *          Unordered map that has a key - value pair of gate name (string) and the function which applies the required noise operators
 *      seed : const unsigned int
 *          Unique seed value for the trajectory to setup the RNG engine
 * 
 * Returns:
 *      std::vector<complex128>
 *          Probabilities of the statevector from the trajectory evolution.
 */
std::vector<complex128> single_trajectory(const std::list<ItemEntry>& instruction_list, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)>& gate_map, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, cuint)>& noise_function_map, cuint seed){
    std::mt19937_64 trajectory_engine(seed);
    std::vector<complex128> trajectory_state(std::size_t{1} << num_qubits, 0.0);
    trajectory_state[0] = 1.0;
    for (const ItemEntry& instruction : instruction_list){
        const std::string& gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, 1);
        std::vector<matrix> noise_matrix_list = get_matrix_list_for_instruction(gate_name, single_qubit_instructions, two_qubit_instructions, qubits[0], qubits[1]);
        noise_function_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, noise_matrix_list, trajectory_engine, 1);
    }
    for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
        const complex128 amplitude = trajectory_state[i];
        trajectory_state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
    }
    return trajectory_state;
}

/**
 * Function that performs the Monte-Carlo Wavefunction Method to approximate the density matrix simulation of a given quantum circuit with a noise instruction
 * 
 * Inputs:
 *      instruction_list : const std::list<ItemEntry>
 *          The list of instructions that are required to build the quantum circuit. Each entry in the list is a struct of ItemEntry
 *      state : complex128*
 *          Pointer to the statevector (or in this case the pointer to the vector that houses the probabilities from the different trajectories)
 *      single_qubit_instructions : const std::vector<noise_map>&
 *          Reference to the set of single qubit noise instructions
 *      two_qubit_instructions : const noise_map2q&
 *          Reference to the set of two qubit noise instructions
 *      num_qubits : const std::size_t
 *          Total number of qubits in the system
 *      num_trajectories : const int
 *          Total number of trajectories to simulate
 *      thread_count : const unsigned int
 *          Total number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
void monte_carlo_wavefunction_solver(const std::list<ItemEntry> instruction_list, complex128* __restrict__ state, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, const int num_trajectories, cuint thread_count){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)> gate_map = gate_function_mapper();
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&, cuint)> noise_function_map = noise_function_mapper();
    double averaging_factor = 1 / static_cast<double>(num_trajectories);
    #pragma omp parallel for shared(instruction_list, num_qubits, single_qubit_instructions, two_qubit_instructions, gate_map, noise_function_map) schedule(dynamic) num_threads(thread_count)
    for (unsigned short t = 0; t < num_trajectories; t++){
        cuint seed = static_cast<cuint>(42 + t);
        std::vector<complex128> trajectory_state = single_trajectory(instruction_list, single_qubit_instructions, two_qubit_instructions, num_qubits, gate_map, noise_function_map, seed);
        #pragma omp critical
        {
            for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
                state[i] += (trajectory_state[i] * averaging_factor);
            }
        }
    }
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
    std::list<ItemEntry> instruction_list;
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
            entry.unitary_matrix = U;
            entry.params = -1.0;
            std::reverse(apply_to_qubits.begin(), apply_to_qubits.end());
            entry.qubits = apply_to_qubits;
        }
        else {
            entry.unitary_matrix = {};
            entry.qubits = apply_to_qubits;
            entry.params = item_tuple[2].cast<double>();
        }
        instruction_list.push_back(entry);
    }
    py::buffer_info state_info = statevector.request();
    auto * state = static_cast<complex128*>(state_info.ptr);
    if (noisy){
        std::vector<noise_map> single_qubit_instructions = parse_single_qubit_noise(single_qubit_noise_instructions);
        noise_map2q two_qubit_instructions = parse_two_qubit_noise(two_qubit_noise_instructions);
        monte_carlo_wavefunction_solver(instruction_list, state, single_qubit_instructions, two_qubit_instructions, num_qubits, num_trajectories, thread_count);
    }
    else {
        unsigned short max_usable_threads = std::min<unsigned short>(thread_count, num_qubits);
        max_usable_threads = std::min<unsigned short>(max_usable_threads, omp_get_max_threads());
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
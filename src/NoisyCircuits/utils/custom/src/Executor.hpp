#pragma once
#include "Communicator.hpp"
#include "TypeDefs.hpp"
#include "NoiseParser.hpp"
#include "QuantumGates.hpp"
#include "NoiseApplicationMPI.hpp"


void set_num_threads(cuint num_threads){
    omp_set_num_threads(num_threads);
}

/*
 * Function that maps the gate name to the function that applies the gate to the state
 * 
 * Inputs:
 *      None
 * 
 * Returns:
 *      std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, const unsigned int)
 *          Map whose key is the name of the gate as a string and the return is the function call to apply the gate.
 */
static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)> gate_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)> gate_map;
    gate_map["x"] = apply_X_gate;
    gate_map["sx"] = apply_SX_gate;
    gate_map["rz"] = apply_RZ_gate;
    gate_map["rx"] = apply_RX_gate;
    gate_map["cz"] = apply_CZ_gate;
    gate_map["ecr"] = apply_ECR_gate;
    gate_map["rzz"] = apply_RZZ_gate;
    gate_map["unitary"] = apply_unitary_gate;
    gate_map["h"] = apply_H_gate;
    gate_map["cx"] = apply_CX_gate;
    gate_map["ry"] = apply_RY_gate;
    gate_map["p"] = apply_P_gate;
    gate_map["swap"] = apply_SWAP_gate;
    return gate_map;
}

/*
 * Function that maps the gate to the correct noise function applicator.
 * 
 * Inputs:
 *      None
 * 
 * Returns:
 *      std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std:size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&)
 *          Map whose key is the gate name as a string and the return is the noise application function.
 */
static inline std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&)> noise_function_mapper(){
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&)> noise_map;
    std::list<std::string> single_qubit_gate_names = {
        "x",
        "sx",
        "rz",
        "rx",
        "h",
        "ry",
        "p"
    };
    std::list<std::string> two_qubit_gate_names = {
        "cx",
        "cz",
        "rzz",
        "cx",
        "swap"
    };
    for (const std::string& gate_name : single_qubit_gate_names){
        noise_map[gate_name] = apply_single_qubit_noise;
    }
    for (const std::string& gate_name : two_qubit_gate_names){
        noise_map[gate_name] = apply_two_qubit_noise;
    }
    noise_map["unitary"] = apply_noise_for_unitary_matrix;
    return noise_map;
}

std::vector<complex128> run_single_trajectory(const std::list<ItemEntry>& instruction_list, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t num_qubits, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)>& gate_map, std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&)> noise_function_map, cuint seed, cuint num_threads){
    std::mt19937_64 trajectory_engine(seed);
    std::vector<complex128> trajectory_state(std::size_t{1} << num_qubits, 0.0);
    trajectory_state[0] = complex128(1.0, 0.0);

    for (const ItemEntry& instruction : instruction_list){
        const std::string gate_name = instruction.gate_name;
        const std::vector<std::size_t>& qubits = instruction.qubits;
        const double params = instruction.params;
        const matrix& unitary_matrix = instruction.unitary_matrix;
        gate_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, params, unitary_matrix, qubits, num_threads);
        std::vector<matrix> noise_instructions = get_matrix_list_for_instruction(gate_name, single_qubit_instructions, two_qubit_instructions, qubits[0], qubits[1]);
        noise_function_map[gate_name](trajectory_state.data(), qubits[0], qubits[1], num_qubits, noise_instructions, trajectory_engine);
    }
    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t{1} << num_qubits); i++){
        const complex128 amplitude = trajectory_state[i];
        trajectory_state[i] = complex128(amplitude.real() * amplitude.real() + amplitude.imag() * amplitude.imag(), 0.0);
    }
    return trajectory_state;
}

std::vector<double> run_trajectories_in_rank(std::list<ItemEntry>& instruction_list, std::vector<noise_map>& single_qubit_instructions, noise_map2q two_qubit_instructions, std::size_t num_qubits, int trajectories_on_rank, cuint num_threads, int rank){
    set_num_threads(num_threads);
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const double, const matrix&, const std::vector<std::size_t>&, cuint)> gate_map = gate_function_mapper();
    std::unordered_map<std::string, void(*)(complex128* __restrict__, const std::size_t, const std::size_t, const std::size_t, const std::vector<matrix>&, std::mt19937_64&)> noise_function_map = noise_function_mapper();
    int base_seed = 42 + rank * trajectories_on_rank;
    std::size_t dim = std::size_t{1} << num_qubits;
    std::vector<double> local_sum(dim, 0.0);
    for (int traj = 0; traj < trajectories_on_rank; ++traj){
        std::vector<complex128> trajectory_result = run_single_trajectory(instruction_list, single_qubit_instructions, two_qubit_instructions, num_qubits, gate_map, noise_function_map, base_seed + traj, num_threads);
        #pragma omp parallel for
        for (std::size_t j = 0; j < dim; ++j){
            local_sum[j] += trajectory_result[j].real();
        }
    }
    return local_sum;
}


std::vector<double> execute(std::list<ItemEntry>& instruction_list, std::vector<noise_map>& single_qubit_instructions, noise_map2q two_qubit_instructions, std::size_t num_qubits, int total_trajectories, cuint num_threads, int root, MPI_Comm comm){
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);
    broadcast_all(instruction_list, single_qubit_instructions, two_qubit_instructions, root, comm);
    std::vector<int> trajectory_counts;
    if (rank == root){
        trajectory_counts = distribute_trajectory(total_trajectories, num_ranks);
    }
    int my_trajectories = scatter_trajectory_count(trajectory_counts, root, comm);
    std::vector<double> local_sum = run_trajectories_in_rank(instruction_list, single_qubit_instructions, two_qubit_instructions, num_qubits, my_trajectories, num_threads, rank);
    std::vector<double> total_sum = reduce_final_result(local_sum, root, comm);
    if (rank == root && total_trajectories > 0){
        double divide_val = 1.0 / total_trajectories;
        #pragma omp parallel for shared(divide_val)
        for (std::size_t t = 0; t < total_sum.size(); ++t){
            total_sum[t] *= divide_val;
        }
    }
    return total_sum;
}
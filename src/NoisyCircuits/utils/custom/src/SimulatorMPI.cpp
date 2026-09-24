/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This source code provides the functionality required to execute quantum circuit simulations from python through the NoisyCircuits library within a distributed memory environment.
 */

#include "Executor.hpp"
#include "TypeDefs.hpp"
#include "NoiseParser.hpp"


static inline std::list<ItemEntry> parse_instructions(py::list instructions){
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
            for (std::size_t i = 0; i < dim; ++i){
                #pragma GCC unroll 2
                for (std::size_t j = 0; j < dim; ++j){
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
    return instruction_list;
}

static inline std::vector<noise_map> get_single_qubit_instructions(py::dict single_qubit_instructions){
    std::vector<noise_map> noise_instructions = parse_single_qubit_noise(single_qubit_instructions);
    return noise_instructions;
}

static inline noise_map2q get_two_qubit_instructions(py::dict two_qubit_instructions){
    noise_map2q noise_instructions = parse_two_qubit_noise(two_qubit_instructions);
    return noise_instructions;
}

py::array_t<double> to_numpy(std::vector<double>&& values){
    auto* held = new std::vector<double>(std::move(values));
    py::capsule owner(held, [](void* p){
        delete static_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        static_cast<py::ssize_t>(held->size()),
        held->data(),
        owner
    );
}

py::array_t<double> simulate_circuit(py::list instructions, py::dict single_qubit_instructions, py::dict two_qubit_instructions, std::size_t num_qubits, int total_trajectories, cuint num_threads){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::list<ItemEntry> instruction_list;
    std::vector<noise_map> single_qubit_noise_instructions;
    noise_map2q two_qubit_noise_instructions;
    if (rank == 0){
        instruction_list = parse_instructions(instructions);
        single_qubit_noise_instructions = get_single_qubit_instructions(single_qubit_instructions);
        two_qubit_noise_instructions = get_two_qubit_instructions(two_qubit_instructions);
    }
    std::vector<double> final_result;
    {
        py::gil_scoped_release release;
        final_result = execute(instruction_list, single_qubit_noise_instructions, two_qubit_noise_instructions, num_qubits, total_trajectories, num_threads, 0, MPI_COMM_WORLD);
    }
    return to_numpy(std::move(final_result));
}

PYBIND11_MODULE(simulator_mpi, m){
    m.doc() = "Module for simulating quantum circuits with noise using the Monte-Carlo Wavefunction method in an MPI Setup.";

    m.def("initialize_mpi", [](){
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized){
            int initialization = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &initialization);
        }
    }, "Initialize MPI with safety in-built for multiple calls to initialization");

    m.def("finalize_mpi", [](){
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized){
            MPI_Finalize();
        }
    }, "Close MPI session with safety built in for multiple calls");

    m.def("get_mpi_rank", [](){
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }, "Get the rank of the process");

    m.def("get_mpi_size", [](){
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }, "Get the total number of ranks in the MPI Session");

    m.def("simulate_circuit", &simulate_circuit, "Run the MCWF Simulation in an MPI Session", py::arg("instructions"), py::arg("single_qubit_instructions"), py::arg("two_qubit_instructions"), py::arg("num_qubits"), py::arg("total_trajectories"), py::arg("num_threads"));
}
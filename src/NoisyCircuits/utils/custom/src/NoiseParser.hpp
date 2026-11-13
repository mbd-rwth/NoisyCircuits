/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This header file is responsible for Parsing and storing qubit noise operators from python to C++ objects.
 */

#pragma once
#include "TypeDefs.hpp"

/**
 * Function that converts a numpy array from python to a C++ usable matrix
 * 
 * Inputs:
 *      array_object : const py::array&
 *          Reference to the numpy array
 * 
 * Returns:
 *      matrix
 *          The matrix representing the noise operator retreived from the numpy array.
 */
static inline matrix to_matrix(const py::array& array_object){
    auto arr = py::array_t<complex128, py::array::c_style | py::array::forcecast>(array_object);
    auto a = arr.unchecked<2>();
    matrix noise_matrix(a.shape(0), std::vector<complex128>(a.shape(1)));
    for (unsigned short i = 0; i < a.shape(0); i++){
        #pragma GCC unroll 2
        for (unsigned short j = 0; j < a.shape(1); j++){
            noise_matrix[i][j] = a(i, j);
        }
    }
    return noise_matrix;
};

/**
 * Function that parses single qubit noise instructions from a python dictionary.
 * 
 * Inputs:
 *      noise_dictionary : const py::dict&
 *          Reference to the python dictionary containing single qubit noise instructions.
 * 
 * Returns: 
 *      std::vector<noise_map>
 *          A vector of an unordered map of gate-names and Kraus operators where the index of the vector represents the qubit number.
 * 
 * Notes:
 *      See _post_process_single_qubit_errors method in BuildQubitGateModel.py and __init__ of QuantumCircuit.py for more information on the structure of the dictionary. The qubit numbers must be in ascending order for this processing to work.
 */
static inline std::vector<noise_map> parse_single_qubit_noise(const py::dict& noise_dictionary){
    std::vector<noise_map> noise_list;
    for (auto item : noise_dictionary){
        int qubit = py::cast<int>(item.first);
        py::dict gate_noise_dict = py::cast<py::dict>(item.second);
        noise_map gate_noise_map;
        for (auto gate_item : gate_noise_dict){
            std::string gate_name = py::cast<std::string>(gate_item.first);
            py::list matrix_list = py::cast<py::list>(gate_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            gate_noise_map.emplace(gate_name, std::move(noise_matrix_list));
        }
        noise_list.push_back(std::move(gate_noise_map));
    }
    return noise_list;
}

/**
 * Function that parses two qubit noise instructions from a python dictionary.
 * 
 * Inputs:
 *      noise_dictionary : const py::dict&
 *          Reference to the python dictionary containing two qubit noise instructions.
 * 
 * Returns: 
 *      noise_map2q
 *          An unordered map of keys gate name and value of an unordered map of keys qubit pairs and value the set of noise operators.
 * 
 * Notes:
 *      See _get_two_qubit_gate_noise_operators method in BuildQubitGateModel.py and __init__ of QuantumCircuit.py for more information on the structure of the dictionary.
 */
static inline noise_map2q parse_two_qubit_noise(const py::dict& noise_dictionary){
    noise_map2q noise_list;
    for (auto item : noise_dictionary){
        std::string gate_name = py::cast<std::string>(item.first);
        py::dict qubit_pair_dict = py::cast<py::dict>(item.second);
        std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash> qubit_pair_map;
        for (auto pair_item : qubit_pair_dict){
            std::pair<int, int> qubit_pair = py::cast<std::pair<int, int>>(pair_item.first);
            py::list matrix_list = py::cast<py::list>(pair_item.second);
            std::vector<matrix> noise_matrix_list;
            noise_matrix_list.reserve(py::len(matrix_list));
            for (auto matrix_item : matrix_list){
                noise_matrix_list.push_back(to_matrix(py::cast<py::array>(matrix_item)));
            }
            qubit_pair_map.emplace(qubit_pair, std::move(noise_matrix_list));
        }
        noise_list.emplace(gate_name, std::move(qubit_pair_map));
    }
    return noise_list;
}

/**
 * Function that gets the whole set of noise instructions for a given gate applied on a particular qubit(s)
 * 
 * Inputs:
 *      gate_name : const std::string&
 *          Reference to the gate name
 *      single_qubit_instructions : const std::vector<noise_map>&
 *          Reference to the entire single qubit noise instructions for system
 *      two_qubit_instructions : const noise_map2q&
 *          Reference to the entire two qubit noise instructions for the system
 *      q1 : const std::size_t
 *          Qubit Index 1
 *      q2 : const std::size_t
 *          Qubit Index 2
 * 
 * Returns:
 *      const std::vector<matrix>& 
 *          A reference to the vector of Kraus operators (complex valued matrices) that represent the noise instructions for the applied gate on a qubit or pair of qubits.
 * 
 * Notes:
 *      Two qubit indices are queried for the inputs to maintain consistent flow with single qubit gates having q1 and q2 with the same value.
 */
const std::vector<matrix>& get_matrix_list_for_instruction(const std::string& gate_name, const std::vector<noise_map>& single_qubit_instructions, const noise_map2q& two_qubit_instructions, const std::size_t q1, const std::size_t q2){
    if (q1 == q2){
        return single_qubit_instructions[q1].at(gate_name);
    }
    else {
        std::pair<int, int> key = {q1, q2};
        return two_qubit_instructions.at(gate_name).at(key);
    }
}
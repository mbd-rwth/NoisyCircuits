/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This source code applies the measurement noise to an array of probabilities.
 */

#include "TypeDefs.hpp"
#include <stdexcept>

/**
 * Function that retrieves the single qubit measurement error matrices from a python dictionary.
 * 
 * Inputs:
 *      noise_dictionary : const py::dict&
 *          Noise dictionary from python containing single qubit measurement error matrices.
 * 
 * Returns:
 *      std::vector<real_matrix>
 *          A vector of real valued (double) matrices that contain the measurement error operators.
 * 
 * Notes:
 *      The input dictionary is a python dictionary whose key-value pairs are qubit_number-measurement_noise (see _extract_measurement_errors method in BuildQubitGateModel.py for more information). The dictionary keys must always be in the ascending order of the qubit numbers.
 *      The return is a vector of real matrices that shift the probabilities according to the observed behaviour of quantum hardware. The index of this vector corresponds to the qubit number and the matrix is the measurement noise.
 */
std::vector<real_matrix> get_measurement_error_matrices(const py::dict& noise_dictionary){
    std::vector<real_matrix> noise_matrix_list;
    for (auto item : noise_dictionary){
        int qubit = py::cast<int>(item.first);
        auto noise_matrix = py::cast<py::array>(item.second);
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(noise_matrix);
        auto a = arr.unchecked<2>();
        real_matrix noise_matrix_converted(a.shape(0), std::vector<double>(a.shape(1)));
        for (unsigned short i = 0; i < a.shape(0); i++){
            #pragma GCC unroll 2
            for (unsigned short j = 0; j < a.shape(1); j++){
                noise_matrix_converted[i][j] = a(i, j);
            }
        }
        noise_matrix_list.push_back(std::move(noise_matrix_converted));
    }
    return noise_matrix_list;
}

/**
 * Function that applies the single qubit measurement noise operator inplace to the probabilities of the simulated quantum circuit.
 * 
 * Inputs:
 *      probabilities : double*
 *          Pointer to the probabilities of the quantum circuit
 *      measurement_noise_operator : real_matrix
 *          Single qubit measurement noise operator.
 *      qubit : const int
 *          Qubit index
 *      num_qubits : const std::size_t
 *          Total number of qubits in the circuit.
 *      num_threads : const unsigned int
 *          Number of threads to distribute the computation
 * 
 * Returns:
 *      None
 */
static inline void apply_measurement_error_for_qubit(double * __restrict__ probabilities, real_matrix measurement_noise_operator, const int qubit, const std::size_t num_qubits, cuint num_threads){
    double m00 = measurement_noise_operator[0][0];
    double m01 = measurement_noise_operator[0][1];
    double m10 = measurement_noise_operator[1][0];
    double m11 = measurement_noise_operator[1][1];
    const std::size_t dim = std::size_t{1} << num_qubits;
    const std::size_t stride = std::size_t{1} << qubit;
    #pragma omp parallel for num_threads(num_threads)
    for (std::size_t p = 0; p < (dim >> 1); ++p){
        const std::size_t i = (p & (stride - 1)) | ((p & ~(stride - 1)) << 1);
        const std::size_t j = i | stride;
        const double p0 = probabilities[i];
        const double p1 = probabilities[j];
        probabilities[i] = m00 * p0 + m01 * p1;
        probabilities[j] = m10 * p0 + m11 * p1;
    }
}

/**
 * Function that performs a feasibility check on the given input parameters to prevent segfaults and heap corruption errors.
 * 
 * Inputs:
 *      num_qubits : std::size_t
 *          Number of qubits being measured in the circuit
 *      probabilities_array : py_array_t<double>&
 *          numpy array containing the probabilities of the simulated quantum circuit
 *      qubit_list : std::vector<int>&
 *          Vector containing the indices of the qubits being measured
 *      measurement_matrices : std::vector<real_matrix>&
 *          Vector containing the measurement error matrices for the measured qubits
 * 
 * Throws:
 *      std::invalid_argument
 *          - Probability array does not contain 2^m entries for m measured qubits
 *          - Fewer readout error matrices were retrieved compared to the number of measured qubits
 *          - When a measured qubit index is out of bounds.
 */
static inline void perform_feasibility_checks(std::size_t num_qubits, py::array_t<double>& probabilities_array, std::vector<int>& qubit_list, std::vector<real_matrix>& measurement_matrices){
    const std::size_t dim = std::size_t{1} << num_qubits;
    if (static_cast<std::size_t>(probabilities_array.size()) != dim){
        throw std::invalid_argument("probability array must have 2^m (m = num_qubits / number of measured qubits) entries");
    }
    if (measurement_matrices.size() < qubit_list.size()){
        throw std::invalid_argument("Fewer readout matrices than measured qubits");
    }
    for (int position : qubit_list){
        if (position < 0 || static_cast<std::size_t>(position) >= num_qubits){
            throw std::invalid_argument("qubit position outside allowed bounds.");
        }
    }
}

/**
 * Main function that controls pre-processing noise and applying noise to the probabilities.
 * 
 * Inputs:
 *      probabilities_array : py::array_t<double>
 *          numpy array containing the probabilities of the simulated quantum circuit
 *      measurement_noise : py::dict
 *          Noise dictionary from python containing single qubit measurement error matrices.
 *      qubit_list : py::list
 *          List of qubits that are measured.
 *      num_qubits : std::size_t
 *          The total number of qubits in the circuit
 *      num_threads : const unsigned int
 *          The number of threads to distribute computation
 * 
 * Returns:
 *      None
 */
void apply_measurement_error(py::array_t<double> probabilities_array, py::dict measurement_noise, py::list qubit_list, std::size_t num_qubits, cuint num_threads){
    std::vector<int> qubit_list_cpp;
    for (auto item : qubit_list){
        qubit_list_cpp.push_back(py::cast<int>(item));
    }
    std::vector<real_matrix> measurement_error_matrices  = get_measurement_error_matrices(measurement_noise);
    perform_feasibility_checks(num_qubits, probabilities_array, qubit_list_cpp, measurement_error_matrices);
    py::buffer_info probs_info = probabilities_array.request();
    auto * probabilities = static_cast<double*>(probs_info.ptr);
    for (std::size_t q = 0; q < qubit_list_cpp.size(); q++){
        int qubit_index = qubit_list_cpp[q];
        real_matrix measurement_noise_operator = measurement_error_matrices[q];
        apply_measurement_error_for_qubit(probabilities, measurement_noise_operator, qubit_index, num_qubits, num_threads);
    }
}

/**
 * Binder for syncing C++ code as a shared library to python.
 */
PYBIND11_MODULE(measurement_error_applicator, m){
    m.doc() = "Module that applies measurement error to a probability distribution over quantum states given a noise model.";
    m.def("apply_measurement_error", &apply_measurement_error, "Applies measurement error to a probability distribution over quantum states", py::arg("probabilities_array"), py::arg("measurement_noise"), py::arg("qubit_list"), py::arg("num_qubits"), py::arg("num_threads"));
}
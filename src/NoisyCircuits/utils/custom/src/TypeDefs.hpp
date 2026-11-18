/**
 * This code is part of NoisyCircuits, (C) Sathyamurthy Hegde 2025, 2026
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or at the root directory of this repository.
 */

/**
 * This header file is responsible for library imports and custom data type definitions.
 */

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <list>
#include <unordered_map>
#include <random>
#include <string>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <array>

namespace py = pybind11;

/**
 * Struct necessary to store information on 2 Qubit Noise operators.
 */
struct pair_hash{
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

using complex128 = std::complex<double>;
// Confusing name --> Needs a name change and will be corrected later.
using cuint = const unsigned int;
using matrix = std::vector<std::vector<complex128>>;
using real_matrix = std::vector<std::vector<double>>;
using noise_map = std::unordered_map<std::string, std::vector<matrix>>;
using noise_map2q = std::unordered_map<std::string, std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash>>;

/**
 * Struct for holding information regarding the quantum circuit.
 */
struct ItemEntry{
    std::string gate_name;
    std::vector<std::size_t> qubits;
    double params;
    matrix unitary_matrix;
};
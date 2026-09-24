#pragma once
#include "TypeDefs.hpp"
#include <utility>
#include <mpi.h>
#include <cstddef>


// Communication API

// Broadcast data to all ranks
void broadcast_all(std::list<ItemEntry>& instruction_list, std::vector<noise_map>& single_qubit_instructions, noise_map2q& two_qubit_instructions, int root, MPI_COMM comm);

// Send Trajectory count to all ranks
int scatter_trajectory_count(const std::vector<int>& trajectory_count, int root, MPI_COMM comm);

// Get the sum of all trajectories in a reduction
std::vector<double> reduce_final_result(const std::vector<double>& rank_sum, int root, MPI_COMM comm);

std::vector<int> distribute_trajectory(int total_trajectories, int num_ranks);
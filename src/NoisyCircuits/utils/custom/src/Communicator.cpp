#include "Communicator.hpp"
#include <cstring>
#include <stdexcept>


namespace {

    void append_bytes(std::vector<char>& buf, const void* p, std::size_t n){
        const char* c = static_cast<const char*>(p);
        buf.insert(buf.end(), c, c + n);
    }

    void append_int(std::vector<char>& buf, int v){
        append_bytes(buf, &v, sizeof(int));
    }

    void append_double(std::vector<char>& buf, double v){
        append_bytes(buf, &v, sizeof(double));
    }

    void append_complex(std::vector<char>& buf, const complex128& c){
        append_bytes(buf, &c, sizeof(complex128));
    }

    void append_string(std::vector<char>& buf, const std::string& s){
        append_int(buf, static_cast<int>(s.size()));
        append_bytes(buf, s.data(), s.size());
    }

    void append_size_vector(std::vector<char>& buf, const std::vector<std::size_t>& v){
        append_int(buf, static_cast<int>(v.size()));
        if (!v.empty()) append_bytes(buf, v.data(), v.size() * sizeof(std::size_t));
    }

    void append_matrix(std::vector<char>& buf, const matrix& m){
        append_int(buf, static_cast<int>(m.size()));
        for (const auto& row : m){
            append_int(buf, static_cast<int>(row.size()));
            for (const auto& val : row) append_complex(buf, val);
        }
    }

    void append_matrix_vector(std::vector<char>& buf, const std::vector<matrix>& mats){
        append_int(buf, static_cast<int>(mats.size()));
        for (const auto& m : mats) append_matrix(buf, m);
    }

    // Read Data
    struct Reader{
        const std::vector<char>& buf;
        std::size_t offset = 0;

        void read_bytes(void* dest, std::size_t n){
            std::memcpy(dest, buf.data() + offset, n);
            offset += n;
        }

        int read_int(){
            int v;
            read_bytes(&v, sizeof(int));
            return v;
        }

        double read_double(){
            double v;
            read_bytes(&v, sizeof(double));
            return v;
        }

        complex128 read_complex(){
            complex128 v;
            read_bytes(&v, sizeof(complex128));
            return v;
        }

        std::string read_string(){
            int len = read_int();
            std::string s(len, '\0');
            if (len > 0) read_bytes(&s[0], len);
            return s;
        }

        std::vector<std::size_t> read_size_vector(){
            int count = read_int();
            std::vector<std::size_t> v(count);
            if (count > 0) read_bytes(v.data(), count * sizeof(std::size_t));
            return v;
        }

        matrix read_matrix(){
            int rows = read_int();
            matrix m(rows);
            for (int r = 0; r < rows; ++r){
                int cols = read_int();
                m[r].resize(cols);
                for (int c = 0; c < cols; ++c){
                    m[r][c] = read_complex();
                }
            }
            return m;
        }

        std::vector<matrix> read_matrix_vector(){
            int count = read_int();
            std::vector<matrix> mats(count);
            for (int i = 0; i < count; ++i){
                mats[i] = read_matrix();
            }
            return mats;
        }
    };

    // Pack top-level data

    std::vector<char> pack_items(const std::list<ItemEntry>& instructions){
        std::vector<char> buf;
        append_int(buf, static_cast<int>((instructions.size())));
        for (const ItemEntry& instruction : instructions){
            append_string(buf, instruction.gate_name);
            append_size_vector(buf, instruction.qubits);
            append_double(buf, instruction.params);
            append_matrix(buf, instruction.unitary_matrix);
        }
        return buf;
    }

    void append_noise_map(std::vector<char>& buf, const noise_map& map){
        append_int(buf, static_cast<int>(map.size()));
        for (const auto& [key, mats] : map){
            append_string(buf, key);
            append_matrix_vector(buf, mats);
        }
    }

    std::vector<char> pack_noise_map(const std::vector<noise_map> maps){
        std::vector<char> buf;
        append_int(buf, static_cast<int>(maps.size()));
        for (const auto& map : maps){
            append_noise_map(buf, map);
        }
        return buf;
    }

    std::vector<char> pack_noise_map2q(const noise_map2q map2q){
        std::vector<char> buf;
        append_int(buf, static_cast<int>(map2q.size()));
        for (const auto& [outer_key, inner_map] : map2q){
            append_string(buf, outer_key);
            append_int(buf, static_cast<int>(inner_map.size()));
            for (const auto& [pair_key, mats] : inner_map){
                append_int(buf, pair_key.first);
                append_int(buf, pair_key.second);
                append_matrix_vector(buf, mats);
            }
        }
        return buf;
    }


    // Unpack all objects
    std::list<ItemEntry> unpack_items(const std::vector<char>& raw){
        Reader r{raw};
        std::list<ItemEntry> instructions;
        int count = r.read_int();
        for (int i = 0; i < count; ++i){
            ItemEntry instruction;
            instruction.gate_name = r.read_string();
            instruction.qubits = r.read_size_vector();
            instruction.params = r.read_double();
            instruction.unitary_matrix = r.read_matrix();
            instructions.push_back(instruction);
        }
        return instructions;
    }

    noise_map unpack_noise_map(Reader r){
        noise_map single_qubit_instructions;
        int count = r.read_int();
        for (int i = 0; i < count; ++i){
            std::string key = r.read_string();
            single_qubit_instructions.emplace(std::move(key), r.read_matrix_vector());
        }
        return single_qubit_instructions;
    }

    std::vector<noise_map> unpack_noise_map_vector(const std::vector<char>& raw){
        Reader r{raw};
        int count = r.read_int();
        std::vector<noise_map> single_qubit_instruction_list(count);
        for (int i = 0; i < count; ++i){
            single_qubit_instruction_list[i] = unpack_noise_map(r);
        }
        return single_qubit_instruction_list;
    }

    noise_map2q unpack_noise_map2q(const std::vector<char>& raw){
        Reader r{raw};
        noise_map2q two_qubit_instructions;
        int outer_count = r.read_int();
        for (int i = 0; i < outer_count; ++i){
            std::string outer_key = r.read_string();
            int inner_count = r.read_int();
            std::unordered_map<std::pair<int, int>, std::vector<matrix>, pair_hash> inner;
            for (int j = 0; j < inner_count; ++j){
                int first = r.read_int();
                int second = r.read_int();
                inner.emplace(std::make_pair(first, second), r.read_matrix_vector());
            }
            two_qubit_instructions.emplace(std::move(outer_key), std::move(inner));
        }
        return two_qubit_instructions;
    }

    // Broadcast a single packed buffer
    std::vector<char> bcast_buffer(std::vector<char> buf, int root, MPI_Comm comm){
        int rank;
        MPI_Comm_rank(comm, &rank);
        int size = static_cast<int>(buf.size());
        MPI_Bcast(&size, 1, MPI_INT, root, comm);
        if (rank != root) buf.resize(size);
        MPI_Bcast(buf.data(), size, MPI_BYTE, root, comm);
        return buf;
    }

}


void broadcast_all(std::list<ItemEntry>& instruction_list, std::vector<noise_map>& single_qubit_instructions, noise_map2q& two_qubit_instructions, int root, MPI_Comm comm){
    {
        std::vector<char> buf = bcast_buffer(pack_items(instruction_list), root, comm);
        instruction_list = unpack_items(buf);
    }
    {
        std::vector<char> buf = bcast_buffer(pack_noise_map(single_qubit_instructions), root, comm);
        single_qubit_instructions = unpack_noise_map_vector(buf);
    }
    {
        std::vector<char> buf = bcast_buffer(pack_noise_map2q(two_qubit_instructions), root, comm);
        two_qubit_instructions = unpack_noise_map2q(buf);
    }
}

std::vector<double> reduce_final_result(const std::vector<double>& local_result, int root, MPI_Comm comm){
    int rank;
    MPI_Comm_rank(comm, &rank);
    int local_size = static_cast<int>(local_result.size());
    std::vector<double> result(local_size, 0.0);
    MPI_Reduce(local_result.data(), result.data(), local_size, MPI_DOUBLE, MPI_SUM, root, comm);
    if (rank != root) result.clear();
    return result;
}

std::vector<int> distribute_trajectory(int total_trajectories, int num_ranks){
    if (num_ranks <= 0){
        throw std::runtime_error("Number of available ranks must be positive");
    }
    if (total_trajectories <= 0){
        throw std::runtime_error("Number of trajectories to distribute must be positive");
    }
    int base = total_trajectories / num_ranks;
    int remainder = total_trajectories % num_ranks;
    std::vector<int> counts(num_ranks, base);
    for (int i = 0; i < remainder; ++i){
        counts[i] += 1;
    }
    return counts;
}

int scatter_trajectory_count(const std::vector<int>& trajectory_counts, int root, MPI_Comm comm){
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);
    if (rank == root && static_cast<int>(trajectory_counts.size()) != num_ranks){
        throw std::runtime_error("Not possible to distribute trajectory counts to all ranks");
    }
    int my_trajectories = 0;
    MPI_Scatter(rank == root ? trajectory_counts.data() : nullptr, 1, MPI_INT, &my_trajectories, 1, MPI_INT, root, comm);
    return my_trajectories;
}
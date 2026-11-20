
#pragma once
#include "TypeDefs.hpp"
#include "NoiseApplication.hpp"
#include <atomic>
#include <memory>
#include <thread>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define NC_CPU_RELAX() _mm_pause()
#else
#define NC_CPU_RELAX() ((void)0)
#endif
#include <cstdio>
#include <unistd.h>


struct alignas(64) GroupBarrier{
    std::atomic<unsigned> counter{0};
    alignas(64) std::atomic<unsigned> generate{0};
    alignas(64) unsigned size = 1;
    void wait(){
        if (size == 1){
            return;
        }
        const unsigned gen = generate.load(std::memory_order_acquire);
        if (counter.fetch_add(1, std::memory_order_acq_rel) + 1 == size){
            counter.store(0, std::memory_order_relaxed);
            generate.fetch_add(1, std::memory_order_release);
            return;
        }
        unsigned spins = 0;
        while (generate.load(std::memory_order_acquire) == gen){
            if (++spins < 4096){
                NC_CPU_RELAX();
            }
            else {
                std::this_thread::yield();
            }
        }
    } 
};

static inline std::size_t pair_index(const std::size_t p, const std::size_t st){
    return (p & (st - 1)) | ((p & ~(st - 1)) << 1);
}

static inline std::size_t quad_index(const std::size_t g, const std::size_t low, const std::size_t high){
    const std::size_t g_l = (g & (low - 1)) | ((g & ~(low - 1)) << 1);
    return (g_l & (high - 1)) | ((g_l & ~(high - 1)) << 1);
}

static inline void partial_rho(const complex128* __restrict__ s, const std::size_t num_targets, const std::size_t s1, const std::size_t s2, const std::size_t u0, const std::size_t u1, double* __restrict__ out){
    if (num_targets == 1){
        double r00 = 0.0, r11 = 0.0, rr = 0.0, ri = 0.0;
        std::size_t p = u0;
        while (p < u1){
            const std::size_t end_run = (s1 >= 4) ? std::min(u1, (p | (s1 - 1)) + 1) : p + 1;
            const std::size_t length = end_run - p;
            const double* __restrict__ x = reinterpret_cast<const double*>(s + pair_index(p, s1));
            const double* __restrict__ y = reinterpret_cast<const double*>(s + pair_index(p, s1) + s1);
            #pragma omp simd reduction(+:r00, r11, rr, ri)
            for (std::size_t k = 0; k < length; ++k){
                const double a = x[2 * k], b = x[2 * k + 1], c = y[2 * k], d = y[2 * k + 1];
                r00 += a * a + b * b;
                r11 += c * c + d * d;
                rr += a * c + b * d;
                ri += b * c - a * d;
            }
            p = end_run;
        }
        out[0] = r00;
        out[1] = r11;
        out[2] = rr;
        out[3] = ri;
        return;
    }
    const std::size_t low = std::min(s1, s2), high = std::max(s1, s2), s12 = s1 + s2;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    double r01_r = 0.0, r01_i = 0.0, r02_r = 0.0, r02_i = 0.0, r03_r = 0.0, r03_i = 0.0;
    double r12_r = 0.0, r12_i = 0.0, r13_r = 0.0, r13_i = 0.0, r23_r = 0.0, r23_i = 0.0;
    std::size_t g = u0;
    while (g < u1){
        const std::size_t end_run = (low >= 4) ? std::min(u1, (g | (low - 1)) + 1) : g + 1;
        const std::size_t length = end_run - g;
        const std::size_t i0 = quad_index(g, low, high);
        const double* __restrict__ v0 = reinterpret_cast<const double*>(s + i0);
        const double* __restrict__ v1 = reinterpret_cast<const double*>(s + i0 + s1);
        const double* __restrict__ v2 = reinterpret_cast<const double*>(s + i0 + s2);
        const double* __restrict__ v3 = reinterpret_cast<const double*>(s + i0 + s1 + s2);
        #pragma omp simd reduction(+:d0, d1, d2, d3, r01_r, r01_i, r02_r, r02_i, r03_r, r03_i, r12_r, r12_i, r13_r, r13_i, r23_r, r23_i)
        for (std::size_t k = 0; k < length; ++k){
            const double x0 = v0[2 * k], y0 = v0[2 * k + 1], x1 = v1[2 * k], y1 = v1[2 * k + 1];
            const double x2 = v2[2 * k], y2 = v2[2 * k + 1], x3 = v3[2 * k], y3 = v3[2 * k + 1];
            d0 += x0 * x0 + y0 * y0;
            d1 += x1 * x1 + y1 * y1;
            d2 += x2 * x2 + y2 * y2;
            d3 += x3 * x3 + y3 * y3;
            r01_r += x0 * x1 + y0 * y1;
            r01_i += y0 * x1 - x0 * y1;
            r02_r += x0 * x2 + y0 * y2;
            r02_i += y0 * x2 - x0 * y2;
            r03_r += x0 * x3 + y0 * y3;
            r03_i += y0 * x3 - x0 * y3;
            r12_r += x1 * x2 + y1 * y2;
            r12_i += y1 * x2 - x1 * y2;
            r13_r += x1 * x3 + y1 * y3;
            r13_i += y1 * x3 - x1 * y3;
            r23_r += x2 * x3 + y2 * y3;
            r23_i += y2 * x3 - x2 * y3;
        }
        g = end_run;
    }
    const double v[16] = {d0, d1, d2, d3, r01_r, r01_i, r02_r, r02_i, r03_r, r03_i, r12_r, r12_i, r13_r, r13_i, r23_r, r23_i};
    std::copy(v, v+ 16, out);
}

static inline void rho_from_sum(const std::size_t num_targets, const double* v, complex128* rho){
    if (num_targets == 1){
        rho[0] = complex128(v[0], 0.0);
        rho[1] = complex128(v[2], v[3]);
        rho[2] = complex128(v[2], -v[3]);
        rho[3] = complex128(v[1], 0.0);
        return;
    }
    rho[0] = complex128(v[0], 0.0);
    rho[1] = complex128(v[4], v[5]);
    rho[2] = complex128(v[6], v[7]);
    rho[3] = complex128(v[8], v[9]);
    rho[4] = complex128(v[4], -v[5]);
    rho[5] = complex128(v[1], 0.0);
    rho[6] = complex128(v[10], v[11]);
    rho[7] = complex128(v[12], v[13]);
    rho[8] = complex128(v[6], -v[7]);
    rho[9] = complex128(v[10], -v[11]);
    rho[10] = complex128(v[2], 0.0);
    rho[11] = complex128(v[14], v[15]);
    rho[12] = complex128(v[8], -v[9]);
    rho[13] = complex128(v[12], -v[13]);
    rho[14] = complex128(v[14], -v[15]);
    rho[15] = complex128(v[3], 0.0);
}

static inline void select_operator(const KrausChannel& channel, const complex128* __restrict__ rho, const complex128* __restrict__ gate, std::mt19937_64& engine, complex128* __restrict__ A){
    const std::size_t d = std::size_t{1} << channel.num_targets;
    std::vector<double> probs_list;
    get_kraus_probs(channel, rho, probs_list);
    std::discrete_distribution<> dist(probs_list.begin(), probs_list.end());
    const std::size_t c = dist(engine);
    matrix_mult(channel.kraus.data() + c * d * d, gate, A, d);
    const double one_by_sqrt_p = 1.0 / std::sqrt(probs_list[c]);
    for (std::size_t e = 0; e < d * d; ++e){
        A[e] *= one_by_sqrt_p;
    }
}

static inline void apply_range(complex128* __restrict__ s, const std::size_t num_targets, const std::size_t s1, const std::size_t s2, const std::size_t u0, const std::size_t u1, const complex128* __restrict__ A){
    if (num_targets == 1){
        const double a00_r = A[0].real(), a00_i = A[0].imag(), a01_r = A[1].real(), a01_i = A[1].imag();
        const double a10_r = A[2].real(), a10_i = A[2].imag(), a11_r = A[3].real(), a11_i = A[3].imag();
        std::size_t p = u0;
        while (p < u1){
            const std::size_t end_run = (s1 >= 4) ? std::min(u1, (p | (s1 - 1)) + 1) : p + 1;
            const std::size_t length = end_run - p;
            double* __restrict__ x = reinterpret_cast<double*>(s + pair_index(p, s1));
            double* __restrict__ y = reinterpret_cast<double*>(s + pair_index(p, s1) + s1);
            #pragma omp simd
            for (std::size_t k = 0; k < length; ++k){
                const double a = x[2 * k], b = x[2 * k + 1], c = y[2 * k], d = y[2 * k + 1];
                x[2 * k] = a00_r * a - a00_i * b + a01_r * c - a01_i * d;
                x[2 * k + 1] = a00_r * b + a00_i * a + a01_r * d + a01_i * c;
                y[2 * k] = a10_r * a - a10_i * b + a11_r * c - a11_i * d;
                y[2 * k + 1] = a10_r * b + a10_i * a + a11_r * d + a11_i * c;
            }
            p = end_run;
        }
        return;
    }
    double Ar[16], Ai[16];
    for (int e = 0; e < 16; ++e){
        Ar[e] = A[e].real();
        Ai[e] = A[e].imag();
    }
    const std::size_t low = std::min(s1, s2), high = std::max(s1, s2), s12 = s1 + s2;
    std::size_t g = u0;
    while (g < u1){
        const std::size_t end_run = (low >= 4) ? std::min(u1, (g | (low - 1)) + 1) : g + 1;
        const std::size_t length = end_run - g;
        const std::size_t i0 = quad_index(g, low, high);
        double* __restrict__ v0 = reinterpret_cast<double*>(s + i0);
        double* __restrict__ v1 = reinterpret_cast<double*>(s + i0 + s1);
        double* __restrict__ v2 = reinterpret_cast<double*>(s + i0 + s2);
        double* __restrict__ v3 = reinterpret_cast<double*>(s + i0 + s12);
        #pragma omp simd
        for (std::size_t k = 0; k < length; ++k){
            const double xr[4] = {v0[2 * k], v1[2 * k], v2[2 * k], v3[2 * k]};
            const double xi[4] = {v0[2 * k + 1], v1[2 * k + 1], v2[2 * k + 1], v3[2 * k + 1]};
            double yr[4], yi[4];
            for (int r = 0; r < 4; ++r){
                yr[r] = Ar[4 * r] * xr[0] - Ai[4 * r] * xi[0] + Ar[4 * r + 1] * xr[1] - Ai[4 * r + 1] * xi[1] + Ar[4 * r + 2] * xr[2] - Ai[4 * r + 2] * xi[2] + Ar[4 * r + 3] * xr[3] - Ai[4 * r + 3] * xi[3];
                yi[r] = Ar[4 * r] * xi[0] + Ai[4 * r] * xr[0] + Ar[4 * r + 1] * xi[1] + Ai[4 * r + 1] * xr[1] + Ar[4 * r + 2] * xi[2] + Ai[4 * r + 2] * xr[2] + Ar[4 * r + 3] * xi[3] + Ai[4 * r + 3] * xr[3];
            }
            v0[2 * k] = yr[0];
            v1[2 * k] = yr[1];
            v2[2 * k] = yr[2];
            v3[2 * k] = yr[3];
            v0[2 * k + 1] = yi[0];
            v1[2 * k + 1] = yi[1];
            v2[2 * k + 1] = yi[2];
            v3[2 * k + 1] = yi[3];
        }
        g = end_run;
    }
}

static inline unsigned read_count_from_list(const char* path){
    FILE* f = std::fopen(path, "r");
    if (f == nullptr){
        return 0;
    }
    unsigned count = 0, a = 0, b = 0;
    char sep = 0;
    while (std::fscanf(f, "%u", &a) == 1){
        b = a;
        if (std::fscanf(f, "%c", &sep) == 1 && sep == '-'){
            if (std::fscanf(f, "%u", &b) != 1){
                break;
            }
            if (std::fscanf(f, "%c", &sep) != 1){
                sep = 0;
            }
        }
        count += b - a + 1;
        if (sep != ','){
            break;
        }
    }
    std::fclose(f);
    return count;
}

static inline unsigned auto_group_size(const std::size_t num_qubits, const unsigned thread_count){
    const long l3_bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
    const unsigned l3_cpus = read_count_from_list("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list");
    unsigned domain_cores = read_count_from_list("/sys/devices/system/node/node0/cpulist");
    if (l3_bytes <= 0 || l3_cpus == 0){
        return 1;
    }
    if (domain_cores == 0 || domain_cores > thread_count){
        domain_cores = thread_count;
    }
    const double per_core_l3 = static_cast<double>(l3_bytes) / l3_cpus;
    const double working_set = 24.0 * static_cast<double>(std::size_t{1} << num_qubits);
    const unsigned needed = static_cast<unsigned>(std::ceil(working_set / per_core_l3));
    for (unsigned gs = std::max(1u, needed); gs <= domain_cores; ++gs){
        if (domain_cores % gs == 0){
            return gs;
        }
    }
    return domain_cores;
}

template<typename NoiseInstructionT>
static void monte_carlo_wavefunction_grouped(const std::vector<ItemEntry>& instruction_list, const std::vector<GateFunction>& gate_functions, const std::vector<NoiseInstructionT>& noise_instructions, complex128* __restrict__ state, const std::size_t num_qubits, const unsigned num_trajectories, const unsigned thread_count, unsigned group_size){
    const std::size_t dim = std::size_t{1} << num_qubits;
    group_size = std::max(1u, std::min(group_size, thread_count));
    const unsigned num_groups = thread_count / group_size;
    std::unique_ptr<GroupBarrier[]> barriers(new GroupBarrier[num_groups]);
    for (unsigned g = 0; g < num_groups; ++g){
        barriers[g].size = group_size;
    }
    constexpr std::size_t SLOT = 32;
    std::unique_ptr<double[]> partial_storage(new double[std::size_t{num_groups} * group_size * SLOT + 8]);
    double* partials = reinterpret_cast<double*>((reinterpret_cast<std::uintptr_t>(partial_storage.get()) + 63) & ~std::uintptr_t{63});
    std::vector<std::vector<complex128>> states(num_groups);
    std::vector<std::vector<double>> accumulators(num_groups);
    const double averaging_factor = 1.0 / static_cast<double>(num_trajectories);
    #pragma omp parallel num_threads(thread_count)
    {
        const unsigned tid = omp_get_thread_num();
        const unsigned g = tid / group_size;
        const unsigned r = tid % group_size;
        if (g < num_groups){
            GroupBarrier& barrier = barriers[g];
            if (r == 0){
                states[g].assign(dim, complex128(0.0, 0.0));
                accumulators[g].assign(dim, 0.0);
            }
            barrier.wait();
            complex128* __restrict__ s = states[g].data();
            double* __restrict__ acc = accumulators[g].data();
            double* group_partials = partials + std::size_t{g} * group_size * SLOT;
            double* my_partial = group_partials + std::size_t{r} * SLOT;
            const std::size_t lo = dim * r / group_size, hi = dim * (r + 1) / group_size;
            for (unsigned t = g; t < num_trajectories; t += num_groups){
                std::mt19937_64 engine(static_cast<unsigned int>(42 + t));
                std::fill(s + lo, s + hi, complex128(0.0, 0.0));
                if (lo == 0){
                    s[0] = complex128(1.0, 0.0);
                }
                for (std::size_t gi = 0; gi < instruction_list.size(); ++gi){
                    barrier.wait();
                    const ItemEntry& instruction = instruction_list[gi];
                    const std::vector<std::size_t>& qubits = instruction.qubits;
                    const std::size_t q2 = second_qubit(qubits);
                    const KrausChannel& channel = *noise_instructions[gi].channel;
                    if (channel.num_targets == 0){
                        if (r == 0){
                            gate_functions[gi](s, qubits[0], q2, num_qubits, instruction.params, instruction.unitary_matrix, qubits, 1);
                        }
                        continue;
                    }
                    const std::size_t nt = channel.num_targets;
                    const std::size_t s1 = std::size_t{1} << qubits[0];
                    const std::size_t s2 = std::size_t{1} << q2;
                    const std::size_t units = dim >> nt;
                    const std::size_t u0 = units * r / group_size, u1 = units * (r + 1) / group_size;
                    partial_rho(s, nt, s1, s2, u0, u1, my_partial);
                    barrier.wait();
                    double sums[16] = {0.0};
                    const std::size_t nv = (nt == 1) ? 4 : 16;
                    for (unsigned m = 0; m < group_size; ++m){
                        const double* p = group_partials + std::size_t{m} * SLOT;
                        for (std::size_t e = 0; e < nv; ++e){
                            sums[e] += p[e];
                        }
                    }
                    complex128 rho[16], A[16];
                    rho_from_sum(nt, sums, rho);
                    update_reduced_density_matrix(rho, noise_instructions[gi].gate.data(), std::size_t{1} << nt);
                    select_operator(channel, rho, noise_instructions[gi].gate.data(), engine, A);
                    apply_range(s, nt, s1, s2, u0, u1, A);
                }
                barrier.wait();
                #pragma omp simd
                for (std::size_t i = lo; i < hi; ++i){
                    acc[i] += s[i].real() * s[i].real() + s[i].imag() * s[i].imag();
                }
            }
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < dim; ++i){
            double sum = 0.0;
            for (unsigned g2 = 0; g2 < num_groups; ++g2){
                sum += accumulators[g2][i];
            }
            state[i] += complex128(sum * averaging_factor, 0.0);
        }
    }
}
#include "benchmark_registerer.hh"
#include "fixture.hh"
#include "main_helper.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

#include <rmm/mr/device/pool_memory_resource.hpp>

int main(int argc, char** argv)
{
    // Google bench setup
    using benchmark_t = benchmark::internal::Benchmark;
    ::benchmark::Initialize(&argc, argv);

    // RMM Setup
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    bool bench_nsight = parse_arguments(argc, argv);

    // Benchmarks registration
    Fixture fx;
    {
        //Sizes to check correctness
        constexpr std::array sizes = {
            31, //All possible block sizes +/-1, includes small odd numbers
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            8, //Small even numbers that are not powers of two
            34,
            130,
            260,
            516,
            1020,
            1024*2, //Forced to cascade with the max block size
            1024*3, //Forced to cascade with the max block size leading to an uneven amount of block
            1024*1025, //Forced to cascade multiple times
            1024*1024*1024-1, //Odd large number
            1024*1024*1024+1, //Odd large number
            1024*1024*1024, //Even large number that is a power of two
            1024*1024*1024-2, //Even large number that is not a power of two
        };

        //To bench
        // constexpr std::array sizes = {
        //     1024*1024*1024
        // };

        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple reduce_to_bench{
            // "base",
            // &base,
            // "less_warp_divergence",
            // &less_warp_divergence,
            // "no_bank_conflict",
            // &no_bank_conflict,
            // "more_work_per_thread",
            // &more_work_per_thread, 
            // "unroll_last_warp",
            // &unroll_last_warp,
            // "unroll_everything",
            // &unroll_everything,
            // "cascading",
            // &cascading,
            // "better_warp_reduce",
            // &better_warp_reduce,
            // "atomics",
            // &atomics,
            // "no_shared",
            // &no_shared,
            // "vectorized",
            // &vectorized,
            "deterministic",
            &deterministic,
        };

        //  / 2 because we store name + function pointer
        benchmark_t* b[tuple_length(reduce_to_bench) / 2];
        int function_index = 0;

        // Call to registerer
        registerer_reduce(&fx,
                          b,
                          function_index,
                          sizes,
                          bench_nsight,
                          reduce_to_bench);
    }
    ::benchmark::RunSpecifiedBenchmarks();
}

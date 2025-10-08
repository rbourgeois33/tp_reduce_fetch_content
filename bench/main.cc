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
            31,
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
            1025, //Forced to cascade
            1024*2, 
            1024*3,
            1024*1025,
            1048576,
            1048576*1,
            1048576*2,
            1048576*3,
            1048576*4,
            1048576*10,
            1048576*100,
            1048576*1003,
            1048576*2000,

        };

        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple reduce_to_bench{
            //"baseline_reduce",
            //&baseline_reduce,
            // "base",
            // &base,
            // "less_warp_divergence",
            // &less_warp_divergence,
            // "no_bank_conflict",
            // &no_bank_conflict,
            //"more_work_per_thread",
            // &more_work_per_thread, 
            "unroll_last_warp",
            &unroll_last_warp,
            //"unroll_everything",
            //&unroll_everything,
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

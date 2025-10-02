#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    for (int i = 0; i < buffer.size(); ++i)
        *total.data() += buffer[i];
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    sdata[tid] = (i >= size) ? 0:buffer[i];

    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        if (tid%(2*s)==0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid==0) block_total[blockIdx.x]=sdata[0];

}
// __global__ void kernel_print(raft::device_span<const int> buffer)
// {
//     unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

//     if (tid >= buffer.size()) return;

//     printf("i = %u, value = %d\n", tid, static_cast<int>(buffer[tid]));
// }

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    const unsigned int BLOCK_SIZE = 64;
    const unsigned int MAX_SIZE_LAST_REDUCE = 128;

    if (MAX_SIZE_LAST_REDUCE>1024) return;

    int size = buffer.size();

    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
    rmm::device_uvector<int> block_total(NBLOCKS, buffer.stream());
    

    bool first_done = false;

    while (size > MAX_SIZE_LAST_REDUCE){

        kernel_your_reduce<int><<<NBLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
            raft::device_span<const int>((first_done) ? block_total.data():buffer.data(), (first_done) ? block_total.size():buffer.size()),
            raft::device_span<int>(block_total.data(), block_total.size()),
            size);

            size = NBLOCKS;
            NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
            first_done = true;
    }

    unsigned int size_last_reduce = (size%2==0) ? size:size+1;

    kernel_your_reduce<int><<<1, size_last_reduce, size_last_reduce*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(first_done ? block_total.data():buffer.data(), first_done ? block_total.size():buffer.size()),
        raft::device_span<int>(total.data(), 1),
        size);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
    
}
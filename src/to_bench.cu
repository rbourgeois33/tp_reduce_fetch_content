#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

//Block size for the first reduction loop
constexpr unsigned int BLOCK_SIZE = 1024;

//We stop the cascade when whe have BLOCK_SIZE elements left or less
const unsigned int MAX_SIZE_LAST_REDUCE = BLOCK_SIZE;

//For clarity
constexpr unsigned int WARP_SIZE = 32;

// template <typename T>
// __global__
// void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
// {
//     for (int i = 0; i < buffer.size(); ++i)
//         *total.data() += buffer[i];
// }

// void baseline_reduce(rmm::device_uvector<int>& buffer,
//                      rmm::device_scalar<int>& total)
// {
// 	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream()>>>(
//         raft::device_span<int>(buffer.data(), buffer.size()),
//         raft::device_span<int>(total.data(), 1));

//     CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
// }


template <typename T>
__global__
void kernel_base(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    
    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
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

// template <typename T>
// __global__
// void kernel_less_warp_divergence(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
// {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
//     sdata[tid] = (i >= size) ? 0:buffer[i];
//     __syncthreads();
//     for (int s=1; s<blockDim.x; s*=2){
//         int index = 2*s*tid;
//         if (index < blockDim.x){
//             sdata[index] += sdata[index+s];
//         }
//         __syncthreads();
//     }
//     if (tid==0) block_total[blockIdx.x]=sdata[0];
// }

// template <typename T>
// __global__
// void kernel_no_bank_conflict(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
// {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
//     sdata[tid] = (i >= size) ? 0:buffer[i];
//     __syncthreads();
//     for (int s=blockDim.x/2; s>0; s>>=1){
//         if (tid<s)
//             sdata[tid] += sdata[tid+s];
//         __syncthreads();
//     }
//     if (tid==0) block_total[blockIdx.x]=sdata[0];
// }

// template <typename T>
// __global__
// void kernel_more_work_per_thread(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
// {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
//     sdata[tid] = (i >= size) ? 0: buffer[i];
//     sdata[tid] += (i+blockDim.x >= size) ? 0: buffer[i+blockDim.x];

//     __syncthreads();
//     for (int s=blockDim.x/2; s>0; s>>=1){
//         if (tid<s)
//             sdata[tid] += sdata[tid+s];
//         __syncthreads();
//     }
//     if (tid==0) block_total[blockIdx.x]=sdata[0];
// }

// // Warp-level reduction (assumes warp size = 32)
// __device__ void warp_reduce(int* sdata, int tid) {
//     if (tid < 32) {sdata[tid] += sdata[tid + 32];}  __syncthreads();
//     if (tid < 16) {sdata[tid] += sdata[tid + 16];}  __syncthreads();
//     if (tid < 8) {sdata[tid] += sdata[tid + 8];}  __syncthreads();
//     if (tid < 4) {sdata[tid] += sdata[tid + 4];}   __syncthreads();
//     if (tid < 2) {sdata[tid] += sdata[tid + 2];}   __syncthreads();
//     if (tid < 1) {sdata[tid] += sdata[tid + 1];}   __syncthreads();
// }

// template <typename T>
// __global__
// void kernel_unroll_last_warp(raft::device_span<const T> buffer, raft::device_span<T> block_total, const int size)
// {
//     extern __shared__ int sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;
//     sdata[tid] = (i >= size) ? 0: buffer[i];
//     sdata[tid] += (i+blockDim.x >= size) ? 0: buffer[i+blockDim.x];

//     __syncthreads();
//     for (int s=blockDim.x/2; s>WARP_SIZE; s>>=1){
//         if (tid<s)
//             sdata[tid] += sdata[tid+s];
//         __syncthreads();
//     }

//     warp_reduce(sdata, tid);
    
//     if (tid==0) block_total[blockIdx.x]=sdata[0];
// }



template <typename KernelFunc, typename T>
void reduce_template( KernelFunc kernel,
                 rmm::device_uvector<T>& buffer,
                 rmm::device_scalar<T>& total)
{
    int size = buffer.size();

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

    //Intermediate arrays to store intermediate reduce result, 2 to avoid race condition
    rmm::device_uvector<T> block_total_in(NBLOCKS, buffer.stream());
    rmm::device_uvector<T> block_total_out(NBLOCKS, buffer.stream());

    //Bool that checks if we have done at least one cascade.
    bool first_done = false;

    // While the amount of element to reduce (size) is > to the max size of the last reduction
    while (size > MAX_SIZE_LAST_REDUCE){

        //We perform a reduction and get one value per block
        //First argument (input of the reduction) is buffer for the first pass, else it's block_total_in
        //Second argument (output) is always block_total_out
        kernel<<<NBLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
            raft::device_span<const int>((first_done) ? block_total_in.data():buffer.data(), (first_done) ? block_total_in.size():buffer.size()),
            raft::device_span<int>(block_total_out.data(), block_total_out.size()),
            size);
        
        //The new amount of element to reduce is NBLOCK
        size = NBLOCKS;
        //We compute the new amount of blocks that we need
        NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

        //We have done at least one pass
        first_done = true;
        
        //Input becomes output to avoid race condition
        std::swap(block_total_out, block_total_in);
    }


    //The last reduction must be done on a even size, even if we have an odd amount of values to reduce
    //The kernel will safely fill out of bound values with 0 in the shared memory
    unsigned int size_last_reduce = (size%2==0) ? size:size+1;

    //We launch 1 block of size size_last_reduce
    //We check if we performed one cascade, because if we did not, we need to input buffer, not block_total_in 
    //the last output of the cascade
    kernel<<<1, size_last_reduce, size_last_reduce*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(first_done ? block_total_in.data():buffer.data(), first_done ? block_total_in.size():buffer.size()),
        raft::device_span<int>(total.data(), 1),
        size);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

void base(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total){
    reduce_template(kernel_base<int>,buffer, total);
}

void less_warp_divergence(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_less_warp_divergence<int>,buffer, total);
}

void no_bank_conflict(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_no_bank_conflict<int>,buffer, total);
}

void more_work_per_thread(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_more_work_per_thread<int>,buffer, total);
}

void unroll_last_warp(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_unroll_last_warp<int>, buffer, total);
}
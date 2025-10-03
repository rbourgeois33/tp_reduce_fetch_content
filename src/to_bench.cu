#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

//Block size for the first reduction loop
constexpr unsigned int BLOCK_SIZE = 256;
//We stop the cascade when whe have BLOCK_SIZE elements left or less
const unsigned int MAX_SIZE_LAST_REDUCE = BLOCK_SIZE;
//For clarity
constexpr unsigned int WARP_SIZE = 32;

__global__ void kernel_print(raft::device_span<int> buffer, int size)
{
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid > size) return;
    printf("i = %u, value = %d\n", tid, static_cast<int>(buffer[tid]));
}

__global__
void kernel_base(raft::device_span<const int> buffer, raft::device_span<int> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        if (tid%(2*s)==0){
            assert((tid+s<blockDim.x)); // Check not out of bound
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) block_total[blockIdx.x]=sdata[0];
}

__global__
void kernel_less_warp_divergence(raft::device_span<const int> buffer, raft::device_span<int> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        int index = 2*s*tid;
        if (index < blockDim.x){
            assert(index+s<blockDim.x); // Check not out of bound
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
    if (tid==0) block_total[blockIdx.x]=sdata[0];
}

__global__
void kernel_no_bank_conflict(raft::device_span<const int> buffer, raft::device_span<int> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){
            assert((tid+s<blockDim.x)); // Check not out of bound
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) block_total[blockIdx.x]=sdata[0];
}

__global__
void kernel_more_work_per_thread(raft::device_span<const int> buffer, raft::device_span<int> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    sdata[tid] += (i+blockDim.x < size) ? buffer[i+blockDim.x]:0;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){
            assert((tid+s<blockDim.x)); // Check not out of bound
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) block_total[blockIdx.x]=sdata[0];
}

__device__ void warp_reduce(int* sdata, int tid) {
    
    sdata[tid] += sdata[tid + 32];  __syncthreads();
    sdata[tid] += sdata[tid + 16];  __syncthreads();
    sdata[tid] += sdata[tid + 8];   __syncthreads();
    sdata[tid] += sdata[tid + 4];   __syncthreads();
    sdata[tid] += sdata[tid + 2];   __syncthreads();
    sdata[tid] += sdata[tid + 1];   __syncthreads();
}

__global__
void kernel_unroll_last_warp(raft::device_span<const int> buffer, raft::device_span<int> block_total, const int size)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    sdata[tid] += (i+blockDim.x < size) ? buffer[i+blockDim.x]:0;
    __syncthreads();

    for (int s=blockDim.x/2; s>32; s>>=1){
        if (tid<s){
            assert((tid+s<blockDim.x)); // Check not out of bound
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid<32) warp_reduce(sdata, tid);

    if (tid==0) block_total[blockIdx.x]=sdata[0];
}


template <typename KernelFunc>
void reduce_template( KernelFunc kernel,
                 rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

    //Intermediate arrays to store intermediate reduce result, 2 to avoid race condition
    rmm::device_uvector<int> block_total_in(NBLOCKS, buffer.stream());
    rmm::device_uvector<int> block_total_out(NBLOCKS, buffer.stream());

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

    assert(size<=1024); 
    assert(size<=MAX_SIZE_LAST_REDUCE); 

    //if (buffer.size()==513) kernel_print<<<1,size>>>(raft::device_span<int>(block_total_in.data(), block_total_in.size()), size);

    //Le dessin marche que si la taille de bloc est une puissance de 2. On veut la puissance de 2 la plus petite qui est > size
    //On démarre à 32 (plus petit warp)
    unsigned int block_size_last_reduce = 64;
    while (block_size_last_reduce < size) block_size_last_reduce*=2;
    
    assert(block_size_last_reduce<=1024); //Pas plus gros que max
    assert(block_size_last_reduce<=MAX_SIZE_LAST_REDUCE); //Pas plus gros que la taille max qu'on s'est fixés
    assert((block_size_last_reduce & (block_size_last_reduce - 1)) == 0); //Puissance de 2

    //We launch 1 block of size block_size_last_reduce
    //We check if we performed one cascade, because if we did not, we need to input buffer, not block_total_in 
    //the last output of the cascade
    kernel<<<1, block_size_last_reduce, block_size_last_reduce*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(first_done ? block_total_in.data():buffer.data(), first_done ? block_total_in.size():buffer.size()),
        raft::device_span<int>(total.data(), 1),
        size);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

void base(rmm::device_uvector<int>& buffer,
          rmm::device_scalar<int>& total){
    reduce_template(kernel_base,buffer, total);
}

void less_warp_divergence(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_less_warp_divergence,buffer, total);
}

void no_bank_conflict(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_no_bank_conflict,buffer, total);
}

void more_work_per_thread(rmm::device_uvector<int>& buffer,
            rmm::device_scalar<int>& total){
     reduce_template(kernel_more_work_per_thread,buffer, total);
}

void unroll_last_warp(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_unroll_last_warp, buffer, total);
}


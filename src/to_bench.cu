#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

//Block size for the reductions
constexpr unsigned int BLOCK_SIZE = 256;
//For clarity
constexpr unsigned int WARP_SIZE = 32;

__global__ void kernel_print(raft::device_span<int> buffer, int size)
{
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid > size) return;
    printf("i = %u, value = %d\n", tid, static_cast<int>(buffer[tid]));
}

__global__
void kernel_base(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

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
    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

__global__
void kernel_less_warp_divergence(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

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
    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

__global__
void kernel_no_bank_conflict(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

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
    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

__global__
void kernel_more_work_per_thread(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);
    
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;

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
    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

__device__ void warp_reduce(int* sdata, int tid) {
    
    //Corner case to be careful about: si la block size c'est 32, on veut pas faire le s=32 (car s=16 !!!)
    if constexpr(BLOCK_SIZE>32) {if (tid<32) {sdata[tid] += sdata[tid + 32];  __syncwarp();}}

    if (tid<16) {sdata[tid] += sdata[tid + 16];  __syncwarp();}
    if (tid<8) {sdata[tid] += sdata[tid + 8];  __syncwarp();}
    if (tid<4) {sdata[tid] += sdata[tid + 4];  __syncwarp();}
    if (tid<2) {sdata[tid] += sdata[tid + 2];  __syncwarp();}
    if (tid<1) {sdata[tid] += sdata[tid + 1];  __syncwarp();}
}

__global__
void kernel_unroll_last_warp(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;

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

    warp_reduce(sdata, tid);

    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}


__global__
void kernel_unroll_everything(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;
    sdata[tid] += (i+blockDim.x < size) ? buffer[i+blockDim.x]:0;
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512){
        if (tid < 256){
            assert((tid+256<blockDim.x));
            sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
    }

    if constexpr (BLOCK_SIZE >= 256){
        if (tid < 128){
            assert((tid+128<blockDim.x));
            sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
    }

    if constexpr (BLOCK_SIZE >= 128){
        if (tid < 64){
            assert((tid+64<blockDim.x));
            sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }
    }

    if (tid<WARP_SIZE) warp_reduce(sdata, tid);

    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}


__global__
void kernel_cascading(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
     //This one I don't get

    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = 0;

    while (i < size){
        sdata[tid] = (i < size) ? buffer[i]:0;
        sdata[tid] += (i+blockDim.x < size) ? buffer[i+blockDim.x]:0;
        i += gridSize;
    }

    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512){
        if (tid < 256){
            assert((tid+256<blockDim.x));
            sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
    }

    if constexpr (BLOCK_SIZE >= 256){
        if (tid < 128){
            assert((tid+128<blockDim.x));
            sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
    }

    if constexpr (BLOCK_SIZE >= 128){
        if (tid < 64){
            assert((tid+64<blockDim.x));
            sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }
    }

    if (tid<WARP_SIZE) warp_reduce(sdata, tid);

    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

template <typename KernelFunc>
void reduce_template( KernelFunc kernel,
                 rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

    //Intermediate arrays to store intermediate reduce result, 2 to avoid race condition
    rmm::device_uvector<int> result_per_block_in(NBLOCKS, buffer.stream());
    rmm::device_uvector<int> result_per_block_out(NBLOCKS, buffer.stream());

    //Bool that checks if we have done at least one cascade.
    bool first_done = false;

    //We stop the cascade when he have BLOCK_SIZE elements left or less
    while (size > BLOCK_SIZE){

        //We perform a reduction and get one value per block
        //First argument (input of the reduction) is buffer for the first pass, else it's result_per_block_in
        //Second argument (output) is always result_per_block_out
        kernel<<<NBLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
            raft::device_span<const int>((first_done) ? result_per_block_in.data():buffer.data(), (first_done) ? result_per_block_in.size():buffer.size()),
            raft::device_span<int>(result_per_block_out.data(), result_per_block_out.size()),
            size);
        
        //The new amount of element to reduce is NBLOCK
        size = NBLOCKS;
        //We compute the new amount of blocks that we need
        NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

        //We have done at least one pass
        first_done = true;
        
        //Input becomes output to avoid race condition
        std::swap(result_per_block_out, result_per_block_in);
    }

    assert(size<=1024); 
    assert(size<=BLOCK_SIZE); 

    //if (buffer.size()==513) kernel_print<<<1,size>>>(raft::device_span<int>(result_per_block_in.data(), result_per_block_in.size()), size);

    //We launch 1 block of size BLOCK_SIZE
    //Not size because the drawing works only is the block size is a power of 2
    //We check if we performed one cascade, because if we did not, we need to input buffer, not result_per_block_in the last output of the cascade
    kernel<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(first_done ? result_per_block_in.data():buffer.data(), first_done ? result_per_block_in.size():buffer.size()),
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

void unroll_everything(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_unroll_everything, buffer, total);
}

 //This one I don't get
void cascading(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_cascading, buffer, total);
}
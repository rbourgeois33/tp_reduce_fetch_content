#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <cuda/atomic>

//Block size for the reductions
constexpr unsigned int BLOCK_SIZE = 256;
//For clarity
constexpr unsigned int WARP_SIZE = 32;

//Util
__global__ void kernel_print(raft::device_span<int> buffer, int size)
{
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid > size) return;
    printf("i = %u, value = %d\n", tid, static_cast<int>(buffer[tid]));
}

//Optim 1-7 below

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
    //We load two values now, so divide NBLOCK by 2 (less_factor=2)
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
    //We load two values now, so divide NBLOCK by 2 (less_factor=2)
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
    //We load two values now, so divide NBLOCK by 2 (less_factor=2)
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

    //We load any arbitrary number of values now, but at least 2, less_factor has to be >=2
    while (i < size){
        sdata[tid] += buffer[i]+ ((i+blockDim.x < size) ? buffer[i+blockDim.x]:0);
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


inline __device__ int better_warp_reduce(int val) {
    
    #define FULL_MASK 0xffffffff
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(~0, val, offset);
    
    return val;
}


__global__
void kernel_better_warp_reduce(raft::device_span<const int> buffer, raft::device_span<int> result_per_block, const int size)
{
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

    //We load any arbitrary number of values now, but at least 2, less_factor has to be >=2
    while (i < size){
        sdata[tid] += buffer[i]+ ((i+blockDim.x < size) ? buffer[i+blockDim.x]:0);
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

    //We need to add this since the better warp reduce does, well the warp reduction
    if constexpr (BLOCK_SIZE >= 64){
        if (tid < 32){
            assert((tid+32<blockDim.x));
            sdata[tid] += sdata[tid + 32];
            __syncthreads();
        }
    }

    sdata[tid] = better_warp_reduce(sdata[tid]);

    if (tid==0) result_per_block[blockIdx.x]=sdata[0];
}

//Template for launching kernel 1-7
template <typename KernelFunc>
void reduce_template( KernelFunc kernel,
                 rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total,
                 const int less_block=1)
{
    //Last argument less_block is the reduction factor of the number of blocks.
    //Base value is 1
    //It's has to be two starting with "more work per threads"
    //And can be more starting with "algo cascading"

    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE*less_block-1)/(BLOCK_SIZE*less_block);

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
        NBLOCKS=(size+BLOCK_SIZE*less_block-1)/(BLOCK_SIZE*less_block);

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









//Below, optim 8 and beyond, because another template is needed









__global__
void kernel_atomics(raft::device_span<const int> buffer, raft::device_span<int> total, const int size)
{
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

    //We load any arbitrary number of values now, but at least 2, less_factor has to be >=2
    while (i < size){
        sdata[tid] += buffer[i]+ ((i+blockDim.x < size) ? buffer[i+blockDim.x]:0);
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

    //We need to add this since the better warp reduce does, well the warp reduction
    if constexpr (BLOCK_SIZE >= 64){
        if (tid < 32){
            assert((tid+32<blockDim.x));
            sdata[tid] += sdata[tid + 32];
            __syncthreads();
        }
    }


    sdata[tid] = better_warp_reduce(sdata[tid]);
    
    //Device sync
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(*total.data());

    if (tid==0) ref.fetch_add(sdata[0], cuda::memory_order_relaxed);

}

__global__
void kernel_no_shared(raft::device_span<const int> buffer, raft::device_span<int> total, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2+threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;

    //We load any arbitrary number of values now, but at least 2, less_factor has to be >=2
    int sum = 0;
    while (i < size){
        sum += buffer[i]+ ((i+blockDim.x < size) ? buffer[i+blockDim.x]:0);
        i += gridSize;
    }

    sum = better_warp_reduce(sum);
    
    //Device sync
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(*total.data());

    if (tid % WARP_SIZE==0) ref.fetch_add(sum, cuda::memory_order_relaxed);
}

__global__
void kernel_vectorized(raft::device_span<const int> buffer, raft::device_span<int> total, const int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE*gridDim.x;

    int sum = 0;
    int bound = (size+4-1)/4; //to not miss the end with vecto
    while (i < bound){
        const int4 val = reinterpret_cast<const int4*>(buffer.data())[i];
        //safer out of bound check
        const int I=4*i;
        sum += ((I < size) ? val.x:0) + ((I+1 < size) ? val.y:0) + ((I+2 < size) ?val.z:0) + ((I+3 < size) ? val.w:0);
        i += gridSize;
    }

    sum = better_warp_reduce(sum);
    
    //Device sync
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(*total.data());

    if (tid % WARP_SIZE==0) ref.fetch_add(sum, cuda::memory_order_relaxed);
}


//Kernel for optim 8-10
template <typename KernelFunc>
void reduce_atomics_template( KernelFunc kernel,
                 rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total,
                 int less_block=1,
                 bool no_shared = false)
                 {
    //Last argument less_block is the reduction factor of the number of blocks.
    //Base value is 1
    //It's has to be two starting with "more work per threads"
    //And can be more starting with "algo cascading"

    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE*less_block-1)/(BLOCK_SIZE*less_block);

    kernel<<<NBLOCKS, BLOCK_SIZE, no_shared ? 0:BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
            raft::device_span<const int>(buffer.data(), buffer.size()),
            raft::device_span<int>(total.data(), 1),
            size);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}




//Declarations





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
     reduce_template(kernel_more_work_per_thread,buffer, total, 2);
    //Last argument is the reduction factor of the number of blocks
    //It's two since each block does twice the job
}

void unroll_last_warp(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_unroll_last_warp, buffer, total, 2);
}

void unroll_everything(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_unroll_everything, buffer, total, 2);
}

void cascading(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_cascading, buffer, total, 8);
    //Last argument is the reduction factor of the number of blocks
    //It's a free parameter now, but has to be >=2 because our algo assumes so
}

void better_warp_reduce(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_template(kernel_better_warp_reduce, buffer, total, 8);
    //Last argument is the reduction factor of the number of blocks
    //It's a free parameter now, but has to be >=2 because our algo assumes so
}

void atomics(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_atomics_template(kernel_atomics, buffer, total, 8);
}

void no_shared(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_atomics_template(kernel_no_shared, buffer, total, 8, true);
    //True for "no shared memory"
}

void vectorized(rmm::device_uvector<int>& buffer,
           rmm::device_scalar<int>& total){
    reduce_atomics_template(kernel_vectorized, buffer, total, 16, true);
    //True for "no shared memory"
}
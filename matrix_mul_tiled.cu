#include "lib.h"


const int SHMEM_SIZE = 1024;

__global__ void matMul(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    int val = 0;

    for (int i = 0; i < n; i += blockDim.x) {
        s_a[threadIdx.y * blockDim.y + threadIdx.x] = a[row * n + (i + threadIdx.x)];
        s_b[threadIdx.y * blockDim.y + threadIdx.x] = b[(i + threadIdx.y) * n + col];

        __syncthreads();

        for(int j = 0; j < blockDim.x; j++) {
            val += s_a[threadIdx.y * blockDim.y + j] * s_b[j * blockDim.y + threadIdx.x];
        }

        __syncthreads();
    }

    c[row * n + col] = val;
}

void matMulCpu(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = 0;
            for (int k = 0; k < n; k++) {
                val += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = val;
        }
    }
}

void initMatRandom(int* x, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j ++) {
            x[i * n + j] = rand() % 50;
        }
    }
}

void errorCheck(int* xExpected, int* xResult, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j ++) {
            assert(xExpected[i * n + j] == xResult[i * n + j]);
        }
    }
    printf("Error check passed, all values are correct\n");
}

int main() {
    const int n = 1024;
    size_t bytes = n * n * sizeof(int);

    int *h_a, *h_b, *h_c, *h_expected;
    int *d_a, *d_b, *d_c;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    h_expected = (int*)malloc(bytes);

    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    initMatRandom(h_a, n);
    initMatRandom(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 NUM_BLOCKS(32, 32);
    dim3 NUM_THREADS((int) ceil((float) n / NUM_BLOCKS.x), (int) ceil((float) n / NUM_BLOCKS.y));
    printf("Total blocks: %d x %d x %d\n", NUM_BLOCKS.x, NUM_BLOCKS.y, NUM_BLOCKS.z);
    printf("Total threads each block: %d x %d x %d\n", NUM_THREADS.x, NUM_THREADS.y, NUM_THREADS.z);

    matMul<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    matMulCpu(h_a, h_b, h_expected, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    errorCheck(h_expected, h_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_expected);

    return 0;
}
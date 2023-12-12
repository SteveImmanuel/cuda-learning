#include "lib.h"

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < n) {
        c[tId] = a[tId] + b[tId];
    }
}

void initRandom(int* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 100;
    }
}

void errorCheck(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
    printf("Error check passed, all values are correct\n");
}

int main() {
    const int n = 1 << 16;
    size_t bytes = n * sizeof(int);

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    initRandom(h_a, n);
    initRandom(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 NUM_THREADS(256);
    dim3 NUM_BLOCKS((int)ceil((float) length / NUM_THREADS.x));
    printf("Total blocks: %d x %d x %d\n", NUM_BLOCKS.x, NUM_BLOCKS.y, NUM_BLOCKS.z);
    printf("Total threads each block: %d x %d x %d\n", NUM_THREADS.x, NUM_THREADS.y, NUM_THREADS.z);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    errorCheck(h_a, h_b, h_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
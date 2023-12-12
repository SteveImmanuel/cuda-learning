#include "lib.h"

__global__ void vectorShiftLeft(int* x, int length) {
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < length - 1) {
        int temp = x[tId + 1];
        __syncthreads();
        x[tId] = temp;
    }
}

void initVector(int* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
}

void printVector(int* x, int n) {
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            printf("%d", x[i]);
        } else {
            printf(" %d", x[i]);
        }
    }
    printf("\n");
}

void errorCheck(int* oriX, int* shiftedX, int n) {
    for (int i = 0; i < n - 1; i++) {
        assert(shiftedX[i] == oriX[i + 1]);
    }
    printf("Error check passed, all values are correct\n");
}

int main() {
    int length = 1000000;
    size_t bytes = length * sizeof(int);

    int* h_x;
    int* h_x_shifted;
    int* d_x;

    h_x = (int*)malloc(bytes);
    h_x_shifted = (int*)malloc(bytes);
    cudaMalloc((void**)&d_x, bytes); 

    initVector(h_x, length);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    dim3 NUM_THREADS(256);
    dim3 NUM_BLOCKS((int)ceil((float) length / NUM_THREADS.x));
    printf("Total blocks: %d x %d x %d\n", NUM_BLOCKS.x, NUM_BLOCKS.y, NUM_BLOCKS.z);
    printf("Total threads each block: %d x %d x %d\n", NUM_THREADS.x, NUM_THREADS.y, NUM_THREADS.z);
    
    vectorShiftLeft<<<NUM_BLOCKS, NUM_THREADS>>>(d_x, length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_x_shifted, d_x, bytes, cudaMemcpyDeviceToHost);
    // printf("Before:\n");
    // printVector(h_x, length);
    // printf("After:\n");
    // printVector(h_x_shifted, length);
    errorCheck(h_x, h_x_shifted, length);

    cudaFree(d_x);
    free(h_x);
    free(h_x_shifted);

    return 0;
}
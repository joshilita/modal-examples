#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Forward declaration of the CUDA kernel
__global__ void invsqrt_kernel(float *out, const float *X, int size);

int main()
{
    int size = 1024;
    float *h_X = (float *)malloc(size * sizeof(float));
    float *h_out = (float *)malloc(size * sizeof(float));
    float *d_X, *d_out;

    // initialize input array with perfect squares
    for (int i = 0; i < size; i++)
    {
        h_X[i] = (float)((i + 1) * (i + 1));
    }

    cudaMalloc((void **)&d_X, size * sizeof(float));
    cudaMalloc((void **)&d_out, size * sizeof(float));

    cudaMemcpy(d_X, h_X, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    invsqrt_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_X, size);

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few of the results
    for (int i = 0; i < 10; i++)
    {
        printf("invsqrt(%f) = %f\n", h_X[i], h_out[i]);
    }

    // Cleanup
    free(h_X);
    free(h_out);
    cudaFree(d_X);
    cudaFree(d_out);

    return 0;
}

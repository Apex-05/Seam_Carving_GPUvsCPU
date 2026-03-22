#include "cuda_kernels.h"

#include <cuda_runtime.h>

#include <float.h>

__global__ static void dp_init_first_row(const float *energy, float *dp, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < width) {
        dp[x] = energy[x];
    }
}

__global__ static void dp_row_kernel(const float *energy, float *dp, int width, int y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) {
        return;
    }

    int row_idx = y * width + x;
    int prev_idx = (y - 1) * width + x;
    float best = dp[prev_idx];

    if (x > 0) {
        float left = dp[prev_idx - 1];
        if (left < best) {
            best = left;
        }
    }

    if (x + 1 < width) {
        float right = dp[prev_idx + 1];
        if (right < best) {
            best = right;
        }
    }

    dp[row_idx] = energy[row_idx] + best;
}

extern "C" int cuda_compute_dp(const float *h_energy, int width, int height, float *h_dp) {
    float *d_energy = NULL;
    float *d_dp = NULL;
    size_t bytes = 0;
    int y = 0;
    int threads = 256;
    int blocks = 0;

    if (!h_energy || !h_dp || width <= 0 || height <= 0) {
        cuda_set_last_error("Invalid inputs to cuda_compute_dp");
        return -1;
    }

    bytes = (size_t)width * (size_t)height * sizeof(float);

    if (cuda_malloc_bytes((void **)&d_energy, bytes) != 0) {
        return -1;
    }

    if (cuda_malloc_bytes((void **)&d_dp, bytes) != 0) {
        cuda_free_ptr(d_energy);
        return -1;
    }

    if (cuda_memcpy_htod(d_energy, h_energy, bytes) != 0) {
        cuda_free_ptr(d_energy);
        cuda_free_ptr(d_dp);
        return -1;
    }

    blocks = (width + threads - 1) / threads;
    dp_init_first_row<<<blocks, threads>>>(d_energy, d_dp, width);
    if (cudaGetLastError() != cudaSuccess) {
        cuda_set_last_error("dp_init_first_row launch failed");
        cuda_free_ptr(d_energy);
        cuda_free_ptr(d_dp);
        return -1;
    }

    for (y = 1; y < height; ++y) {
        dp_row_kernel<<<blocks, threads>>>(d_energy, d_dp, width, y);
        if (cudaGetLastError() != cudaSuccess) {
            cuda_set_last_error("dp_row_kernel launch failed");
            cuda_free_ptr(d_energy);
            cuda_free_ptr(d_dp);
            return -1;
        }
    }

    if (cuda_device_synchronize() != 0) {
        cuda_free_ptr(d_energy);
        cuda_free_ptr(d_dp);
        return -1;
    }

    if (cuda_memcpy_dtoh(h_dp, d_dp, bytes) != 0) {
        cuda_free_ptr(d_energy);
        cuda_free_ptr(d_dp);
        return -1;
    }

    cuda_free_ptr(d_energy);
    cuda_free_ptr(d_dp);
    return 0;
}

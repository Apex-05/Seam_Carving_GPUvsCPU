#include "cuda_kernels.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>

#define BLOCK_W 16
#define BLOCK_H 16
#define TILE_W (BLOCK_W + 2)
#define TILE_H (BLOCK_H + 2)

__device__ static int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ static float gray_from_global(const unsigned char *rgb, int width, int height, int x, int y) {
    int cx = clamp_int(x, 0, width - 1);
    int cy = clamp_int(y, 0, height - 1);
    int idx = (cy * width + cx) * 3;

    float r = (float)rgb[idx + 0];
    float g = (float)rgb[idx + 1];
    float b = (float)rgb[idx + 2];
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

/*
 * Shared-memory Sobel:
 * - each thread computes one output pixel
 * - each block loads a 16x16 tile plus 1-pixel halo on all sides into shared memory
 * - halo caching removes repeated global loads from neighboring pixels
 */
__global__ static void energy_sobel_kernel_shared(const unsigned char *rgb, float *energy, int width, int height) {
    __shared__ float tile[TILE_H][TILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    int gx = x;
    int gy = y;

    /* Center load */
    if (tx < BLOCK_W && ty < BLOCK_H) {
        tile[ty + 1][tx + 1] = gray_from_global(rgb, width, height, gx, gy);
    }

    /* Left/right halo */
    if (tx == 0) {
        tile[ty + 1][0] = gray_from_global(rgb, width, height, gx - 1, gy);
    }
    if (tx == BLOCK_W - 1) {
        tile[ty + 1][BLOCK_W + 1] = gray_from_global(rgb, width, height, gx + 1, gy);
    }

    /* Top/bottom halo */
    if (ty == 0) {
        tile[0][tx + 1] = gray_from_global(rgb, width, height, gx, gy - 1);
    }
    if (ty == BLOCK_H - 1) {
        tile[BLOCK_H + 1][tx + 1] = gray_from_global(rgb, width, height, gx, gy + 1);
    }

    /* Corner halo */
    if (tx == 0 && ty == 0) {
        tile[0][0] = gray_from_global(rgb, width, height, gx - 1, gy - 1);
    }
    if (tx == BLOCK_W - 1 && ty == 0) {
        tile[0][BLOCK_W + 1] = gray_from_global(rgb, width, height, gx + 1, gy - 1);
    }
    if (tx == 0 && ty == BLOCK_H - 1) {
        tile[BLOCK_H + 1][0] = gray_from_global(rgb, width, height, gx - 1, gy + 1);
    }
    if (tx == BLOCK_W - 1 && ty == BLOCK_H - 1) {
        tile[BLOCK_H + 1][BLOCK_W + 1] = gray_from_global(rgb, width, height, gx + 1, gy + 1);
    }

    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    float tl = tile[ty + 0][tx + 0];
    float tc = tile[ty + 0][tx + 1];
    float tr = tile[ty + 0][tx + 2];
    float ml = tile[ty + 1][tx + 0];
    float mr = tile[ty + 1][tx + 2];
    float bl = tile[ty + 2][tx + 0];
    float bc = tile[ty + 2][tx + 1];
    float br = tile[ty + 2][tx + 2];

    float gx_sobel = -tl + tr - 2.0f * ml + 2.0f * mr - bl + br;
    float gy_sobel = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;

    energy[y * width + x] = fabsf(gx_sobel) + fabsf(gy_sobel);
}

extern "C" int cuda_compute_energy(
    const unsigned char *h_rgb,
    int width,
    int height,
    float *h_energy,
    float *elapsed_ms
) {
    unsigned char *d_rgb = NULL;
    float *d_energy = NULL;
    size_t rgb_bytes = 0;
    size_t energy_bytes = 0;
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid;
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;
    float measured_ms = 0.0f;
    cudaError_t err;

    if (!h_rgb || !h_energy || width <= 0 || height <= 0) {
        cuda_set_last_error("Invalid inputs to cuda_compute_energy");
        return -1;
    }

    rgb_bytes = (size_t)width * (size_t)height * 3U;
    energy_bytes = (size_t)width * (size_t)height * sizeof(float);

    if (cuda_malloc_bytes((void **)&d_rgb, rgb_bytes) != 0) {
        return -1;
    }

    if (cuda_malloc_bytes((void **)&d_energy, energy_bytes) != 0) {
        cuda_free_ptr(d_rgb);
        return -1;
    }

    if (cuda_memcpy_htod(d_rgb, h_rgb, rgb_bytes) != 0) {
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventCreate(&start_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventCreate(start) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventCreate(&stop_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventCreate(stop) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    grid = dim3((unsigned int)(width + block.x - 1) / block.x, (unsigned int)(height + block.y - 1) / block.y);

    err = cudaEventRecord(start_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventRecord(start) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    energy_sobel_kernel_shared<<<grid, block>>>(d_rgb, d_energy, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "energy kernel launch failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventRecord(stop_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventRecord(stop) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventSynchronize(stop_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventSynchronize(stop) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventElapsedTime(&measured_ms, start_event, stop_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventElapsedTime failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    if (cuda_memcpy_dtoh(h_energy, d_energy, energy_bytes) != 0) {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventDestroy(start_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventDestroy(start) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cudaEventDestroy(stop_event);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    err = cudaEventDestroy(stop_event);
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cudaEventDestroy(stop) failed: %s", cudaGetErrorString(err));
        cuda_set_last_error(msg);
        cuda_free_ptr(d_rgb);
        cuda_free_ptr(d_energy);
        return -1;
    }

    if (cuda_free_ptr(d_rgb) != 0) {
        cuda_free_ptr(d_energy);
        return -1;
    }

    if (cuda_free_ptr(d_energy) != 0) {
        return -1;
    }

    if (elapsed_ms) {
        *elapsed_ms = measured_ms;
    }

    return 0;
}

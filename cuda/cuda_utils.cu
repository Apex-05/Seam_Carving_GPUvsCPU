#include "cuda_kernels.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

static char g_last_error[512] = "OK";

static int set_cuda_status(cudaError_t err, const char *context) {
    if (err == cudaSuccess) {
        return 0;
    }

    snprintf(
        g_last_error,
        sizeof(g_last_error),
        "%s failed: %s",
        context,
        cudaGetErrorString(err)
    );
    return -1;
}

extern "C" void cuda_set_last_error(const char *msg) {
    if (!msg) {
        strncpy(g_last_error, "Unknown CUDA error", sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
        return;
    }

    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

extern "C" const char *cuda_get_last_error(void) {
    return g_last_error;
}

extern "C" int cuda_malloc_bytes(void **ptr, size_t bytes) {
    return set_cuda_status(cudaMalloc(ptr, bytes), "cudaMalloc");
}

extern "C" int cuda_free_ptr(void *ptr) {
    return set_cuda_status(cudaFree(ptr), "cudaFree");
}

extern "C" int cuda_memcpy_htod(void *dst, const void *src, size_t bytes) {
    return set_cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice");
}

extern "C" int cuda_memcpy_dtoh(void *dst, const void *src, size_t bytes) {
    return set_cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost");
}

extern "C" int cuda_device_synchronize(void) {
    return set_cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

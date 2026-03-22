#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_compute_energy(
    const unsigned char *h_rgb,
    int width,
    int height,
    float *h_energy,
    float *elapsed_ms
);

int cuda_compute_dp(const float *h_energy, int width, int height, float *h_dp);

void cuda_set_last_error(const char *msg);
const char *cuda_get_last_error(void);

int cuda_malloc_bytes(void **ptr, size_t bytes);
int cuda_free_ptr(void *ptr);
int cuda_memcpy_htod(void *dst, const void *src, size_t bytes);
int cuda_memcpy_dtoh(void *dst, const void *src, size_t bytes);
int cuda_device_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif

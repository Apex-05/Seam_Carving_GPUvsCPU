#include "seam.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float gray_at_cpu(const unsigned char *rgb, int width, int height, int x, int y) {
    int cx = clamp_int(x, 0, width - 1);
    int cy = clamp_int(y, 0, height - 1);
    int idx = (cy * width + cx) * 3;

    float r = (float)rgb[idx + 0];
    float g = (float)rgb[idx + 1];
    float b = (float)rgb[idx + 2];

    return 0.299f * r + 0.587f * g + 0.114f * b;
}

int compute_energy_cpu(const unsigned char *rgb, int width, int height, float *energy) {
    int y = 0;

    if (!rgb || !energy || width <= 0 || height <= 0) {
        return -1;
    }

    for (y = 0; y < height; ++y) {
        int x = 0;
        for (x = 0; x < width; ++x) {
            float tl = gray_at_cpu(rgb, width, height, x - 1, y - 1);
            float tc = gray_at_cpu(rgb, width, height, x, y - 1);
            float tr = gray_at_cpu(rgb, width, height, x + 1, y - 1);
            float ml = gray_at_cpu(rgb, width, height, x - 1, y);
            float mr = gray_at_cpu(rgb, width, height, x + 1, y);
            float bl = gray_at_cpu(rgb, width, height, x - 1, y + 1);
            float bc = gray_at_cpu(rgb, width, height, x, y + 1);
            float br = gray_at_cpu(rgb, width, height, x + 1, y + 1);

            float gx = -tl + tr - 2.0f * ml + 2.0f * mr - bl + br;
            float gy = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;

            energy[y * width + x] = fabsf(gx) + fabsf(gy);
        }
    }

    return 0;
}

int compute_dp_cpu(const float *energy, int width, int height, float *dp) {
    int y = 0;

    if (!energy || !dp || width <= 0 || height <= 0) {
        return -1;
    }

    for (y = 0; y < width; ++y) {
        dp[y] = energy[y];
    }

    for (y = 1; y < height; ++y) {
        int x = 0;
        for (x = 0; x < width; ++x) {
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
    }

    return 0;
}

int find_vertical_seam_cpu(const float *dp, int width, int height, int *seam) {
    int y = 0;

    if (!dp || !seam || width <= 0 || height <= 0) {
        return -1;
    }

    {
        int min_col = 0;
        float min_val = dp[(height - 1) * width];
        int x = 0;
        for (x = 1; x < width; ++x) {
            float v = dp[(height - 1) * width + x];
            if (v < min_val) {
                min_val = v;
                min_col = x;
            }
        }
        seam[height - 1] = min_col;
    }

    for (y = height - 2; y >= 0; --y) {
        int prev_col = seam[y + 1];
        int best_col = prev_col;
        float best_val = dp[y * width + prev_col];

        if (prev_col > 0) {
            float left_val = dp[y * width + (prev_col - 1)];
            if (left_val < best_val) {
                best_val = left_val;
                best_col = prev_col - 1;
            }
        }

        if (prev_col + 1 < width) {
            float right_val = dp[y * width + (prev_col + 1)];
            if (right_val < best_val) {
                best_val = right_val;
                best_col = prev_col + 1;
            }
        }

        seam[y] = best_col;
    }

    return 0;
}

int remove_seam_cpu(Image *img, const int *seam) {
    int y = 0;
    int new_width = 0;
    unsigned char *new_data = NULL;

    if (!img || !img->data || !seam || img->width <= 1 || img->height <= 0) {
        return -1;
    }

    new_width = img->width - 1;
    new_data = (unsigned char *)malloc((size_t)new_width * (size_t)img->height * 3U);
    if (!new_data) {
        return -1;
    }

    for (y = 0; y < img->height; ++y) {
        int seam_x = seam[y];
        int old_row_offset = y * img->width * 3;
        int new_row_offset = y * new_width * 3;
        int left_bytes = seam_x * 3;
        int right_pixels = img->width - seam_x - 1;
        int right_bytes = right_pixels * 3;

        if (seam_x < 0 || seam_x >= img->width) {
            free(new_data);
            return -1;
        }

        if (left_bytes > 0) {
            memcpy(new_data + new_row_offset, img->data + old_row_offset, (size_t)left_bytes);
        }

        if (right_bytes > 0) {
            memcpy(
                new_data + new_row_offset + left_bytes,
                img->data + old_row_offset + (seam_x + 1) * 3,
                (size_t)right_bytes
            );
        }
    }

    free(img->data);
    img->data = new_data;
    img->width = new_width;
    return 0;
}

int highlight_seam_cpu(const Image *src, const int *seam, Image *dst) {
    int y = 0;

    if (!src || !src->data || !seam || !dst) {
        return -1;
    }

    if (copy_image(src, dst) != 0) {
        return -1;
    }

    for (y = 0; y < src->height; ++y) {
        int x = seam[y];
        int idx = (y * src->width + x) * 3;

        if (x < 0 || x >= src->width) {
            free_image(dst);
            return -1;
        }

        dst->data[idx + 0] = 255;
        dst->data[idx + 1] = 0;
        dst->data[idx + 2] = 0;
    }

    return 0;
}

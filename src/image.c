#include "image.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int next_token(FILE *fp, char *buf, size_t len) {
    int c = 0;
    size_t i = 0;

    do {
        c = fgetc(fp);
        if (c == '#') {
            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
        }
    } while (isspace(c));

    if (c == EOF) {
        return 0;
    }

    while (c != EOF && !isspace(c)) {
        if (i + 1 < len) {
            buf[i++] = (char)c;
        }
        c = fgetc(fp);
    }

    buf[i] = '\0';
    return i > 0;
}

int load_ppm(const char *path, Image *img) {
    FILE *fp = NULL;
    char token[64];
    int is_binary = 0;
    int i = 0;

    if (!path || !img) {
        return -1;
    }

    memset(img, 0, sizeof(*img));

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    if (!next_token(fp, token, sizeof(token))) {
        fclose(fp);
        return -1;
    }

    if (strcmp(token, "P6") == 0) {
        is_binary = 1;
    } else if (strcmp(token, "P3") == 0) {
        is_binary = 0;
    } else {
        fclose(fp);
        return -1;
    }

    if (!next_token(fp, token, sizeof(token))) {
        fclose(fp);
        return -1;
    }
    img->width = atoi(token);

    if (!next_token(fp, token, sizeof(token))) {
        fclose(fp);
        return -1;
    }
    img->height = atoi(token);

    if (!next_token(fp, token, sizeof(token))) {
        fclose(fp);
        return -1;
    }
    img->maxval = atoi(token);

    if (img->width <= 0 || img->height <= 0 || img->maxval <= 0 || img->maxval > 255) {
        fclose(fp);
        return -1;
    }

    img->data = (unsigned char *)malloc((size_t)img->width * (size_t)img->height * 3U);
    if (!img->data) {
        fclose(fp);
        return -1;
    }

    if (is_binary) {
        int c = fgetc(fp);
        if (c == EOF) {
            free_image(img);
            fclose(fp);
            return -1;
        }

        if (fread(img->data, 1, (size_t)img->width * (size_t)img->height * 3U, fp)
            != (size_t)img->width * (size_t)img->height * 3U) {
            free_image(img);
            fclose(fp);
            return -1;
        }
    } else {
        int total_values = img->width * img->height * 3;
        for (i = 0; i < total_values; ++i) {
            int value = 0;
            if (!next_token(fp, token, sizeof(token))) {
                free_image(img);
                fclose(fp);
                return -1;
            }
            value = atoi(token);
            if (value < 0 || value > img->maxval) {
                free_image(img);
                fclose(fp);
                return -1;
            }
            img->data[i] = (unsigned char)value;
        }
    }

    fclose(fp);
    return 0;
}

int save_ppm(const char *path, const Image *img) {
    FILE *fp = NULL;
    size_t bytes = 0;

    if (!path || !img || !img->data || img->width <= 0 || img->height <= 0) {
        return -1;
    }

    fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->maxval > 0 ? img->maxval : 255);

    bytes = (size_t)img->width * (size_t)img->height * 3U;
    if (fwrite(img->data, 1, bytes, fp) != bytes) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

int copy_image(const Image *src, Image *dst) {
    size_t bytes = 0;

    if (!src || !dst || !src->data || src->width <= 0 || src->height <= 0) {
        return -1;
    }

    dst->width = src->width;
    dst->height = src->height;
    dst->maxval = src->maxval;

    bytes = (size_t)src->width * (size_t)src->height * 3U;
    dst->data = (unsigned char *)malloc(bytes);
    if (!dst->data) {
        return -1;
    }

    memcpy(dst->data, src->data, bytes);
    return 0;
}

int rotate_image_90_cw(const Image *src, Image *dst) {
    int y = 0;

    if (!src || !dst || !src->data || src->width <= 0 || src->height <= 0) {
        return -1;
    }

    dst->width = src->height;
    dst->height = src->width;
    dst->maxval = src->maxval;
    dst->data = (unsigned char *)malloc((size_t)dst->width * (size_t)dst->height * 3U);
    if (!dst->data) {
        return -1;
    }

    for (y = 0; y < src->height; ++y) {
        int x = 0;
        for (x = 0; x < src->width; ++x) {
            int src_idx = (y * src->width + x) * 3;
            int dst_x = src->height - 1 - y;
            int dst_y = x;
            int dst_idx = (dst_y * dst->width + dst_x) * 3;

            dst->data[dst_idx + 0] = src->data[src_idx + 0];
            dst->data[dst_idx + 1] = src->data[src_idx + 1];
            dst->data[dst_idx + 2] = src->data[src_idx + 2];
        }
    }

    return 0;
}

int rotate_image_90_ccw(const Image *src, Image *dst) {
    int y = 0;

    if (!src || !dst || !src->data || src->width <= 0 || src->height <= 0) {
        return -1;
    }

    dst->width = src->height;
    dst->height = src->width;
    dst->maxval = src->maxval;
    dst->data = (unsigned char *)malloc((size_t)dst->width * (size_t)dst->height * 3U);
    if (!dst->data) {
        return -1;
    }

    for (y = 0; y < src->height; ++y) {
        int x = 0;
        for (x = 0; x < src->width; ++x) {
            int src_idx = (y * src->width + x) * 3;
            int dst_x = y;
            int dst_y = src->width - 1 - x;
            int dst_idx = (dst_y * dst->width + dst_x) * 3;

            dst->data[dst_idx + 0] = src->data[src_idx + 0];
            dst->data[dst_idx + 1] = src->data[src_idx + 1];
            dst->data[dst_idx + 2] = src->data[src_idx + 2];
        }
    }

    return 0;
}

void free_image(Image *img) {
    if (!img) {
        return;
    }

    free(img->data);
    img->data = NULL;
    img->width = 0;
    img->height = 0;
    img->maxval = 0;
}

#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

typedef struct {
    int width;
    int height;
    int maxval;
    unsigned char *data;
} Image;

int load_ppm(const char *path, Image *img);
int save_ppm(const char *path, const Image *img);
int copy_image(const Image *src, Image *dst);
int rotate_image_90_cw(const Image *src, Image *dst);
int rotate_image_90_ccw(const Image *src, Image *dst);
void free_image(Image *img);

#endif

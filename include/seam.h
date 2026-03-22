#ifndef SEAM_H
#define SEAM_H

#include "image.h"

int compute_energy_cpu(const unsigned char *rgb, int width, int height, float *energy);
int compute_dp_cpu(const float *energy, int width, int height, float *dp);
int find_vertical_seam_cpu(const float *dp, int width, int height, int *seam);
int remove_seam_cpu(Image *img, const int *seam);
int highlight_seam_cpu(const Image *src, const int *seam, Image *dst);

#endif

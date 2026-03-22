#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int ensure_directory(const char *path) {
    if (!path || !path[0]) {
        return -1;
    }

#ifdef _WIN32
    if (_mkdir(path) == 0) {
        return 0;
    }
#else
    if (mkdir(path, 0755) == 0) {
        return 0;
    }
#endif

    return 0;
}

int file_exists(const char *path) {
    if (!path || !path[0]) {
        return 0;
    }

#ifdef _WIN32
    return _access(path, 0) == 0;
#else
    return access(path, F_OK) == 0;
#endif
}

void print_usage(const char *program) {
    printf(
        "Usage: %s input.ppm output.ppm <num_seams> "
        "[--mode cpu|gpu] [--direction vertical|horizontal] [--interactive] [--visualize-energy]\n",
        program
    );
}

int prompt_continue_or_quit(void) {
    char line[32];

    printf("Press ENTER to continue or q to quit: ");
    if (!fgets(line, sizeof(line), stdin)) {
        return 0;
    }

    if (line[0] == 'q' || line[0] == 'Q') {
        return 1;
    }

    return 0;
}

int normalize_energy_to_grayscale(const float *energy, int count, int *grayscale) {
    int i = 0;
    float min_energy = 0.0f;
    float max_energy = 0.0f;

    if (!energy || !grayscale || count <= 0) {
        return -1;
    }

    min_energy = energy[0];
    max_energy = energy[0];

    for (i = 0; i < count; ++i) {
        if (energy[i] < min_energy) {
            min_energy = energy[i];
        }
        if (energy[i] > max_energy) {
            max_energy = energy[i];
        }
    }

    if (max_energy <= min_energy) {
        for (i = 0; i < count; ++i) {
            grayscale[i] = 0;
        }
        return 0;
    }

    /* Stretch [min_energy, max_energy] into [0, 255] for better contrast. */
    for (i = 0; i < count; ++i) {
        float scaled = ((energy[i] - min_energy) * 255.0f) / (max_energy - min_energy);
        if (scaled < 0.0f) {
            scaled = 0.0f;
        }
        if (scaled > 255.0f) {
            scaled = 255.0f;
        }
        grayscale[i] = (int)(scaled + 0.5f);
    }

    return 0;
}

void save_energy_map(const char *filename, const int *energy, int width, int height) {
    FILE *fp = NULL;
    int y = 0;

    if (!filename || !energy || width <= 0 || height <= 0) {
        return;
    }

    fp = fopen(filename, "w");
    if (!fp) {
        return;
    }

    /* Save as plain-text grayscale PPM for quick debugging and visualization. */
    fprintf(fp, "P3\n%d %d\n255\n", width, height);

    for (y = 0; y < height; ++y) {
        int x = 0;
        for (x = 0; x < width; ++x) {
            int idx = y * width + x;
            int val = energy[idx];
            if (val < 0) {
                val = 0;
            }
            if (val > 255) {
                val = 255;
            }
            fprintf(fp, "%d %d %d ", val, val, val);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

const char *mode_to_string(ComputeMode mode) {
    return mode == MODE_CPU ? "CPU" : "GPU";
}

const char *direction_to_string(SeamDirection direction) {
    return direction == DIRECTION_HORIZONTAL ? "horizontal" : "vertical";
}

static void print_mode_report(FILE *fp, const char *label, const BenchmarkStats *stats) {
    fprintf(fp, "Mode: %s\n", label);
    fprintf(fp, "Energy: %.3f ms\n", stats->energy_ms);
    fprintf(fp, "DP: %.3f ms\n", stats->dp_ms);
    fprintf(fp, "Seam Removal: %.3f ms\n", stats->seam_remove_ms);
    fprintf(fp, "Total: %.3f ms\n", stats->total_ms);
}

int write_benchmark_results(
    const char *path,
    const char *input_path,
    const char *output_path,
    int requested_seams,
    int completed_seams,
    int original_width,
    int original_height,
    int final_width,
    int final_height,
    ComputeMode selected_mode,
    SeamDirection direction,
    const BenchmarkStats *selected_stats,
    const BenchmarkStats *other_stats
) {
    double speedup = 0.0;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        return -1;
    }

    fprintf(fp, "GPU-Accelerated Seam Carving Benchmark\n");
    fprintf(fp, "Input: %s\n", input_path);
    fprintf(fp, "Output: %s\n", output_path);
    fprintf(fp, "Direction: %s\n", direction_to_string(direction));
    fprintf(fp, "Original size: %dx%d\n", original_width, original_height);
    fprintf(fp, "Final size: %dx%d\n", final_width, final_height);
    fprintf(fp, "Requested seams: %d\n", requested_seams);
    fprintf(fp, "Completed seams: %d\n", completed_seams);

    fprintf(fp, "\n--- Benchmark Report ---\n");
    print_mode_report(fp, mode_to_string(selected_mode), selected_stats);
    if (other_stats) {
        print_mode_report(fp, mode_to_string(selected_mode == MODE_GPU ? MODE_CPU : MODE_GPU), other_stats);
        if (selected_mode == MODE_GPU && selected_stats->total_ms > 0.0) {
            speedup = other_stats->total_ms / selected_stats->total_ms;
        } else if (selected_mode == MODE_CPU && other_stats->total_ms > 0.0) {
            speedup = selected_stats->total_ms / other_stats->total_ms;
        }
        fprintf(fp, "Speedup: %.3fx\n", speedup);
    }

    fclose(fp);
    return 0;
}

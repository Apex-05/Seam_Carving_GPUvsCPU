#ifndef UTILS_H
#define UTILS_H

typedef enum {
    MODE_GPU = 0,
    MODE_CPU = 1
} ComputeMode;

typedef enum {
    DIRECTION_VERTICAL = 0,
    DIRECTION_HORIZONTAL = 1
} SeamDirection;

typedef struct {
    double energy_ms;
    double dp_ms;
    double seam_remove_ms;
    double total_ms;
    int completed_seams;
} BenchmarkStats;

double now_ms(void);
int ensure_directory(const char *path);
int file_exists(const char *path);
void print_usage(const char *program);
int prompt_continue_or_quit(void);
int normalize_energy_to_grayscale(const float *energy, int count, int *grayscale);
void save_energy_map(const char *filename, const int *energy, int width, int height);

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
);

const char *mode_to_string(ComputeMode mode);
const char *direction_to_string(SeamDirection direction);

#endif

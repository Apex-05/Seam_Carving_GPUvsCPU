#include "cuda_kernels.h"
#include "image.h"
#include "seam.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *input_path;
    const char *output_path;
    int seams_requested;
    int interactive;
    int visualize_energy;
    ComputeMode mode;
    SeamDirection direction;
} AppConfig;

static int parse_int_arg(const char *text, int *out) {
    char *end = NULL;
    long value = 0;

    if (!text || !out) {
        return -1;
    }

    value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 1000000L) {
        return -1;
    }

    *out = (int)value;
    return 0;
}

static int parse_args(int argc, char **argv, AppConfig *cfg) {
    int i = 0;

    if (!cfg || argc < 4) {
        return -1;
    }

    memset(cfg, 0, sizeof(*cfg));
    cfg->input_path = argv[1];
    cfg->output_path = argv[2];
    cfg->mode = MODE_GPU;
    cfg->direction = DIRECTION_VERTICAL;

    if (parse_int_arg(argv[3], &cfg->seams_requested) != 0) {
        fprintf(stderr, "Invalid <num_seams>: %s\n", argv[3]);
        return -1;
    }

    for (i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--interactive") == 0) {
            cfg->interactive = 1;
        } else if (strcmp(argv[i], "--visualize-energy") == 0) {
            cfg->visualize_energy = 1;
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --mode\n");
                return -1;
            }
            ++i;
            if (strcmp(argv[i], "gpu") == 0) {
                cfg->mode = MODE_GPU;
            } else if (strcmp(argv[i], "cpu") == 0) {
                cfg->mode = MODE_CPU;
            } else {
                fprintf(stderr, "Invalid mode: %s (use cpu|gpu)\n", argv[i]);
                return -1;
            }
        } else if (strcmp(argv[i], "--direction") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --direction\n");
                return -1;
            }
            ++i;
            if (strcmp(argv[i], "vertical") == 0) {
                cfg->direction = DIRECTION_VERTICAL;
            } else if (strcmp(argv[i], "horizontal") == 0) {
                cfg->direction = DIRECTION_HORIZONTAL;
            } else {
                fprintf(stderr, "Invalid direction: %s (use vertical|horizontal)\n", argv[i]);
                return -1;
            }
        } else {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            return -1;
        }
    }

    return 0;
}

static int orient_to_processing_space(Image *img, SeamDirection direction) {
    Image rotated;

    if (direction == DIRECTION_VERTICAL) {
        return 0;
    }

    memset(&rotated, 0, sizeof(rotated));
    if (rotate_image_90_cw(img, &rotated) != 0) {
        return -1;
    }

    free_image(img);
    *img = rotated;
    return 0;
}

static int restore_output_orientation(Image *img, SeamDirection direction) {
    Image rotated;

    if (direction == DIRECTION_VERTICAL) {
        return 0;
    }

    memset(&rotated, 0, sizeof(rotated));
    if (rotate_image_90_ccw(img, &rotated) != 0) {
        return -1;
    }

    free_image(img);
    *img = rotated;
    return 0;
}

static int save_seam_image(const Image *processing_img, const int *seam, int iter, SeamDirection direction) {
    Image seam_view;
    Image seam_out;
    char seam_path[256];

    memset(&seam_view, 0, sizeof(seam_view));
    memset(&seam_out, 0, sizeof(seam_out));

    if (highlight_seam_cpu(processing_img, seam, &seam_view) != 0) {
        return -1;
    }

    if (direction == DIRECTION_HORIZONTAL) {
        if (rotate_image_90_ccw(&seam_view, &seam_out) != 0) {
            free_image(&seam_view);
            return -1;
        }
    } else {
        if (copy_image(&seam_view, &seam_out) != 0) {
            free_image(&seam_view);
            return -1;
        }
    }

    snprintf(seam_path, sizeof(seam_path), "output/seam/seam_%d.ppm", iter);
    if (save_ppm(seam_path, &seam_out) != 0) {
        free_image(&seam_view);
        free_image(&seam_out);
        return -1;
    }

    free_image(&seam_view);
    free_image(&seam_out);
    return 0;
}

static int save_step_image(const Image *processing_img, int iter, SeamDirection direction) {
    Image step_out;
    char step_path[256];

    memset(&step_out, 0, sizeof(step_out));

    if (direction == DIRECTION_HORIZONTAL) {
        if (rotate_image_90_ccw(processing_img, &step_out) != 0) {
            return -1;
        }
    } else {
        if (copy_image(processing_img, &step_out) != 0) {
            return -1;
        }
    }

    snprintf(step_path, sizeof(step_path), "output/step/step_%d.ppm", iter);
    if (save_ppm(step_path, &step_out) != 0) {
        free_image(&step_out);
        return -1;
    }

    free_image(&step_out);
    return 0;
}

static void rotate_energy_90_ccw(const int *src, int src_width, int src_height, int *dst) {
    int y = 0;

    for (y = 0; y < src_height; ++y) {
        int x = 0;
        for (x = 0; x < src_width; ++x) {
            int src_idx = y * src_width + x;
            int dst_x = y;
            int dst_y = src_width - 1 - x;
            int dst_idx = dst_y * src_height + dst_x;
            dst[dst_idx] = src[src_idx];
        }
    }
}

static int save_energy_image(
    const float *energy,
    int width,
    int height,
    int iter,
    SeamDirection direction,
    const char *name_suffix
) {
    int count = width * height;
    int *energy_gray = NULL;
    char energy_path[256];

    if (!energy || width <= 0 || height <= 0) {
        return -1;
    }

    energy_gray = (int *)malloc((size_t)count * sizeof(int));
    if (!energy_gray) {
        return -1;
    }

    if (normalize_energy_to_grayscale(energy, count, energy_gray) != 0) {
        free(energy_gray);
        return -1;
    }

    if (name_suffix && name_suffix[0] != '\0') {
        snprintf(energy_path, sizeof(energy_path), "output/energy/energy_%s.ppm", name_suffix);
    } else {
        snprintf(energy_path, sizeof(energy_path), "output/energy/energy_%d.ppm", iter);
    }
    if (direction == DIRECTION_HORIZONTAL) {
        int out_width = height;
        int out_height = width;
        int *rotated = (int *)malloc((size_t)out_width * (size_t)out_height * sizeof(int));
        if (!rotated) {
            free(energy_gray);
            return -1;
        }
        rotate_energy_90_ccw(energy_gray, width, height, rotated);
        save_energy_map(energy_path, rotated, out_width, out_height);
        free(rotated);
    } else {
        save_energy_map(energy_path, energy_gray, width, height);
    }

    free(energy_gray);
    return 0;
}

static int run_pipeline(
    Image *img,
    int seams_to_remove,
    ComputeMode mode,
    SeamDirection direction,
    int interactive,
    int save_steps,
    int visualize_energy,
    BenchmarkStats *stats
) {
    int i = 0;
    int failed = 0;
    double total_start = 0.0;
    double total_end = 0.0;

    if (!img || !img->data || seams_to_remove < 0 || !stats) {
        return -1;
    }

    memset(stats, 0, sizeof(*stats));

    if (orient_to_processing_space(img, direction) != 0) {
        fprintf(stderr, "Failed to rotate image into processing space\n");
        return -1;
    }

    if (seams_to_remove >= img->width) {
        seams_to_remove = img->width - 1;
    }

    total_start = now_ms();

    for (i = 0; i < seams_to_remove; ++i) {
        int width = img->width;
        int height = img->height;
        float *energy = NULL;
        float *dp = NULL;
        int *seam = NULL;
        double t0 = 0.0;
        double t1 = 0.0;
        float gpu_energy_ms = 0.0f;

        energy = (float *)malloc((size_t)width * (size_t)height * sizeof(float));
        dp = (float *)malloc((size_t)width * (size_t)height * sizeof(float));
        seam = (int *)malloc((size_t)height * sizeof(int));

        if (!energy || !dp || !seam) {
            fprintf(stderr, "Allocation failure at iteration %d\n", i);
            free(energy);
            free(dp);
            free(seam);
            failed = 1;
            break;
        }

        if (mode == MODE_GPU) {
            if (cuda_compute_energy(img->data, width, height, energy, &gpu_energy_ms) != 0) {
                fprintf(stderr, "CUDA energy error: %s\n", cuda_get_last_error());
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }
            stats->energy_ms += (double)gpu_energy_ms;
        } else {
            t0 = now_ms();
            if (compute_energy_cpu(img->data, width, height, energy) != 0) {
                fprintf(stderr, "CPU energy computation failed\n");
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }
            t1 = now_ms();
            stats->energy_ms += t1 - t0;
        }

        /* Export optional per-iteration energy map before DP mutates pipeline state. */
        if (visualize_energy) {
            int one_based_iter = i + 1;
            int should_save_checkpoint = (one_based_iter % 5) == 0;
            int is_last_iter = (i == seams_to_remove - 1);

            if (should_save_checkpoint
                && save_energy_image(energy, width, height, one_based_iter, direction, NULL) != 0) {
                fprintf(stderr, "Energy visualization failed at iteration %d\n", i);
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }

            if (is_last_iter
                && save_energy_image(energy, width, height, one_based_iter, direction, "final") != 0) {
                fprintf(stderr, "Final energy visualization failed at iteration %d\n", i);
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }
        }

        t0 = now_ms();
        if (mode == MODE_GPU) {
            if (cuda_compute_dp(energy, width, height, dp) != 0) {
                fprintf(stderr, "CUDA DP error: %s\n", cuda_get_last_error());
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }
        } else {
            if (compute_dp_cpu(energy, width, height, dp) != 0) {
                fprintf(stderr, "CPU DP computation failed\n");
                free(energy);
                free(dp);
                free(seam);
                failed = 1;
                break;
            }
        }
        t1 = now_ms();
        stats->dp_ms += t1 - t0;

        if (find_vertical_seam_cpu(dp, width, height, seam) != 0) {
            fprintf(stderr, "Seam backtracking failed\n");
            free(energy);
            free(dp);
            free(seam);
            failed = 1;
            break;
        }

        if (save_steps && save_seam_image(img, seam, i, direction) != 0) {
            fprintf(stderr, "Failed to save seam visualization at iteration %d\n", i);
            free(energy);
            free(dp);
            free(seam);
            failed = 1;
            break;
        }

        t0 = now_ms();
        if (remove_seam_cpu(img, seam) != 0) {
            fprintf(stderr, "Seam removal failed\n");
            free(energy);
            free(dp);
            free(seam);
            failed = 1;
            break;
        }
        t1 = now_ms();
        stats->seam_remove_ms += t1 - t0;

        if (save_steps && save_step_image(img, i, direction) != 0) {
            fprintf(stderr, "Failed to save step image at iteration %d\n", i);
            free(energy);
            free(dp);
            free(seam);
            failed = 1;
            break;
        }

        free(energy);
        free(dp);
        free(seam);

        stats->completed_seams = i + 1;

        printf(
            "Iteration %d/%d complete | mode=%s | direction=%s | size=%dx%d\n",
            stats->completed_seams,
            seams_to_remove,
            mode_to_string(mode),
            direction_to_string(direction),
            img->width,
            img->height
        );

        if (interactive && prompt_continue_or_quit()) {
            printf("Interactive stop requested after %d seams.\n", stats->completed_seams);
            break;
        }
    }

    total_end = now_ms();
    stats->total_ms = total_end - total_start;

    if (restore_output_orientation(img, direction) != 0) {
        fprintf(stderr, "Failed to restore output orientation\n");
        return -1;
    }

    return failed ? -1 : 0;
}

int main(int argc, char **argv) {
    AppConfig cfg;
    Image original;
    Image output_img;
    Image bench_gpu_img;
    Image bench_cpu_img;
    BenchmarkStats selected_stats;
    BenchmarkStats gpu_stats;
    BenchmarkStats cpu_stats;
    int max_seams = 0;
    int seams_to_remove = 0;
    int completed_for_benchmark = 0;
    double speedup = 0.0;

    memset(&cfg, 0, sizeof(cfg));
    memset(&original, 0, sizeof(original));
    memset(&output_img, 0, sizeof(output_img));
    memset(&bench_gpu_img, 0, sizeof(bench_gpu_img));
    memset(&bench_cpu_img, 0, sizeof(bench_cpu_img));
    memset(&selected_stats, 0, sizeof(selected_stats));
    memset(&gpu_stats, 0, sizeof(gpu_stats));
    memset(&cpu_stats, 0, sizeof(cpu_stats));

    if (parse_args(argc, argv, &cfg) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (!file_exists(cfg.input_path)) {
        fprintf(stderr, "Input file does not exist: %s\n", cfg.input_path);
        return 1;
    }

    if (load_ppm(cfg.input_path, &original) != 0) {
        fprintf(stderr, "Failed to load valid PPM image: %s\n", cfg.input_path);
        return 1;
    }

    max_seams = cfg.direction == DIRECTION_VERTICAL ? original.width - 1 : original.height - 1;
    if (max_seams <= 0) {
        fprintf(stderr, "Image dimension too small for seam carving\n");
        free_image(&original);
        return 1;
    }

    seams_to_remove = cfg.seams_requested > max_seams ? max_seams : cfg.seams_requested;
    if (seams_to_remove != cfg.seams_requested) {
        fprintf(stderr, "Requested seams exceed valid range. Clamping to %d.\n", seams_to_remove);
    }

    if (copy_image(&original, &output_img) != 0) {
        fprintf(stderr, "Failed to allocate working image\n");
        free_image(&original);
        return 1;
    }

    ensure_directory("output");
    ensure_directory("output/step");
    ensure_directory("output/seam");
    ensure_directory("output/energy");
    ensure_directory("benchmarks");

    if (run_pipeline(
            &output_img,
            seams_to_remove,
            cfg.mode,
            cfg.direction,
            cfg.interactive,
            1,
            cfg.visualize_energy,
            &selected_stats
        ) != 0) {
        free_image(&original);
        free_image(&output_img);
        return 1;
    }

    if (save_ppm(cfg.output_path, &output_img) != 0) {
        fprintf(stderr, "Failed to save output image: %s\n", cfg.output_path);
        free_image(&original);
        free_image(&output_img);
        return 1;
    }

    completed_for_benchmark = selected_stats.completed_seams;

    if (completed_for_benchmark > 0) {
        if (copy_image(&original, &bench_gpu_img) == 0) {
            if (run_pipeline(
                    &bench_gpu_img,
                    completed_for_benchmark,
                    MODE_GPU,
                    cfg.direction,
                    0,
                    0,
                    0,
                    &gpu_stats
                ) != 0) {
                memset(&gpu_stats, 0, sizeof(gpu_stats));
            }
        }

        if (copy_image(&original, &bench_cpu_img) == 0) {
            if (run_pipeline(
                    &bench_cpu_img,
                    completed_for_benchmark,
                    MODE_CPU,
                    cfg.direction,
                    0,
                    0,
                    0,
                    &cpu_stats
                ) != 0) {
                memset(&cpu_stats, 0, sizeof(cpu_stats));
            }
        }
    }

    if (gpu_stats.total_ms > 0.0 && cpu_stats.total_ms > 0.0) {
        speedup = cpu_stats.total_ms / gpu_stats.total_ms;
    }

    printf("\n--- Benchmark Report ---\n");
    printf("Mode: GPU\n");
    printf("Energy: %.3f ms\n", gpu_stats.energy_ms);
    printf("DP: %.3f ms\n", gpu_stats.dp_ms);
    printf("Seam Removal: %.3f ms\n", gpu_stats.seam_remove_ms);
    printf("Total: %.3f ms\n", gpu_stats.total_ms);
    printf("Mode: CPU\n");
    printf("Energy: %.3f ms\n", cpu_stats.energy_ms);
    printf("DP: %.3f ms\n", cpu_stats.dp_ms);
    printf("Seam Removal: %.3f ms\n", cpu_stats.seam_remove_ms);
    printf("Total: %.3f ms\n", cpu_stats.total_ms);
    printf("Speedup: %.3fx\n", speedup);

    if (write_benchmark_results(
            "benchmarks/results.txt",
            cfg.input_path,
            cfg.output_path,
            cfg.seams_requested,
            selected_stats.completed_seams,
            original.width,
            original.height,
            output_img.width,
            output_img.height,
            MODE_GPU,
            cfg.direction,
            &gpu_stats,
            &cpu_stats
        ) != 0) {
        fprintf(stderr, "Failed to write benchmark file\n");
    }

    free_image(&original);
    free_image(&output_img);
    free_image(&bench_gpu_img);
    free_image(&bench_cpu_img);
    return 0;
}

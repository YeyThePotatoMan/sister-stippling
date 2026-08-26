#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

void lloyd_assign_simd(const double *density, int width, int height,
                      const double *px, const double *py, int n,
                      double *sum_x, double *sum_y, double *sum_w) {
    int i, x, y;
    for (i = 0; i < n; i++) {
        sum_x[i] = 0.0;
        sum_y[i] = 0.0;
        sum_w[i] = 0.0;
    }
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            double w = density[(size_t)y * width + x];
            if (w == 0.0)
                continue;
            double best_d = -1.0;
            int best = 0;
            int i = 0;
            __m256d vx = _mm256_set1_pd((double)x);
            __m256d vy = _mm256_set1_pd((double)y);
            for (; i + 4 <= n; i += 4) {
                __m256d pxs = _mm256_loadu_pd(&px[i]);
                __m256d pys = _mm256_loadu_pd(&py[i]);
                __m256d dx = _mm256_sub_pd(vx, pxs);
                __m256d dy = _mm256_sub_pd(vy, pys);
                __m256d d = _mm256_add_pd(_mm256_mul_pd(dx, dx),
                                         _mm256_mul_pd(dy, dy));
                double dd[4];
                _mm256_storeu_pd(dd, d);
                for (int k = 0; k < 4; k++) {
                    int idx = i + k;
                    if (best_d < 0.0 || dd[k] < best_d) {
                        best_d = dd[k];
                        best = idx;
                    }
                }
            }
            for (; i < n; i++) {
                double dx = x - px[i];
                double dy = y - py[i];
                double d = dx * dx + dy * dy;
                if (best_d < 0.0 || d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            sum_x[best] += x * w;
            sum_y[best] += y * w;
            sum_w[best] += w;
        }
    }
}

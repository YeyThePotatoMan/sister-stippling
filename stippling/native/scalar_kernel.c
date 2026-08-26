#include <stdint.h>
#include <stddef.h>

void lloyd_assign_scalar(const double *density, int width, int height,
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
            for (i = 0; i < n; i++) {
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

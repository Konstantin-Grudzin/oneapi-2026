#include "shared_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());

    sycl::queue q(device);

    float* a_sh = sycl::malloc_shared<float>(n * n, q);
    float* b_sh = sycl::malloc_shared<float>(n, q);
    float* x_cur = sycl::malloc_shared<float>(n, q);
    float* x_next = sycl::malloc_shared<float>(n, q);

    q.memcpy(a_sh, a.data(), sizeof(float) * n * n).wait();
    q.memcpy(b_sh, b.data(), sizeof(float) * n).wait();
    q.memset(x_cur, 0, sizeof(float) * n).wait();
    q.memset(x_next, 0, sizeof(float) * n).wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int i = static_cast<int>(id[0]);
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += a_sh[i * n + j] * x_cur[j];
                }
            }
            x_next[i] = (b_sh[i] - sum) / a_sh[i * n + i];
        }).wait();

        bool converged = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(x_next[i] - x_cur[i]) >= accuracy) {
                converged = false;
                break;
            }
        }

        float* tmp = x_cur;
        x_cur = x_next;
        x_next = tmp;

        if (converged) break;
    }

    std::vector<float> result(x_cur, x_cur + n);

    sycl::free(a_sh, q);
    sycl::free(b_sh, q);
    sycl::free(x_cur, q);
    sycl::free(x_next, q);

    return result;
}

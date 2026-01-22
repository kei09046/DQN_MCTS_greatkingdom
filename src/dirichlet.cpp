#include "dirichlet.h"

std::vector<float> sample_dirichlet(int k, float alpha) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::gamma_distribution<float> gamma(alpha, 1.0f);

    std::vector<float> vals(k);
    float sum = 0.0f;

    // Sample Gamma(α, 1)
    for (int i = 0; i < k; i++) {
        float v = gamma(rng);
        vals[i] = v;
        sum += v;
    }

    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            vals[i] /= sum;
    }

    return vals;
}
#include "random.h"

namespace rnd {
	thread_local std::mt19937 gen(std::random_device{}());

	std::vector<int> select_indices(int range, int many) {
		std::vector<int> indices(range);
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), gen);

		std::vector<int> selected_cols(indices.begin(), indices.begin() + many);
		return selected_cols;
	}

	int pick0or1(float zeroProb){
		std::bernoulli_distribution dist(1.0f - zeroProb);
		return dist(gen);
	}

	std::vector<float> sample_dirichlet(int k, float alpha) {
		std::gamma_distribution<float> gamma(alpha, 1.0f);

		std::vector<float> vals(k);
		float sum = 0.0f;

		// Sample Gamma(α, 1)
		for (int i = 0; i < k; i++) {
			float v = gamma(gen);
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
}

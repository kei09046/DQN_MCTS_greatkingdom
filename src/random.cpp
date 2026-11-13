#include "random.h"

namespace rnd {
	thread_local std::mt19937 gen(std::random_device{}());

	std::vector<int> select_indices(int range, int many) {
		std::vector<int> indices(range);
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), gen);

		std::vector<int> selectedCols(indices.begin(), indices.begin() + many);
		return selectedCols;
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

	const std::vector<std::vector<Move>>& neutral_point_classes(){
		static_assert(rowSize == colSize, "symmetry classes assume a square board");
		// canonical point of (r, c)'s class under the 8 rotations / flips : fold onto the top-left quadrant, then r <= c.
		const auto canonical = [](int r, int c){
			r = std::min(r, rowSize - 1 - r);
			c = std::min(c, colSize - 1 - c);
			return std::make_pair(std::min(r, c), std::max(r, c));
		};

		static const std::vector<std::vector<Move>> classes = [&]{
			std::vector<std::pair<int, int>> reps;
			std::vector<std::vector<Move>> cls;
			for(int r=0; r<rowSize; ++r)
				for(int c=0; c<colSize; ++c){
					const auto key = canonical(r, c);
					const auto it = std::find(reps.begin(), reps.end(), key);
					const Move m = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
					if(it == reps.end()){
						reps.push_back(key);
						cls.push_back({m});
					}
					else
						cls[it - reps.begin()].push_back(m);
				}
			return cls;
		}();
		return classes;
	}

	int neutral_setup_count(){
		return static_cast<int>(neutral_point_classes().size()) + 1;
	}

	int neutral_setup_index(const std::vector<Move>& neutrals){
		const auto& classes = neutral_point_classes();
		if(neutrals.empty())
			return static_cast<int>(classes.size());
		for(int k=0; k<static_cast<int>(classes.size()); ++k)
			if(std::find(classes[k].begin(), classes[k].end(), neutrals[0]) != classes[k].end())
				return k;
		return -1; // off the board
	}

	std::vector<Move> sample_neutrals(){
		const auto& classes = neutral_point_classes();
		std::uniform_int_distribution<int> pickClass(0, static_cast<int>(classes.size())); // == size : no neutral
		const int k = pickClass(gen);
		if(k == static_cast<int>(classes.size()))
			return {};
		std::uniform_int_distribution<int> pickPoint(0, static_cast<int>(classes[k].size()) - 1);
		return {classes[k][pickPoint(gen)]};
	}

	float sample_fair_komi(float fairKomi, float noise, float wideProb, float wideNoise){
		if(wideProb > 0.0f && std::bernoulli_distribution(wideProb)(gen))
			noise = wideNoise;
		// uniform around fairKomi, cut at 0.5 (not clamped : that would pile the cut-off mass onto 0.5).
		const float lo = std::max(fairKomi - noise, 0.5f), hi = std::max(fairKomi + noise, lo);
		std::uniform_real_distribution<float> draw(lo, hi);
		const float x = (hi > lo) ? draw(gen) : lo;
		// x.5 values on either side of x; the upper one with probability (x - lower), so the mean stays x.
		const float lower = std::floor(x - 0.5f) + 0.5f;
		std::bernoulli_distribution up(x - lower);
		const float komi = up(gen) ? lower + 1.0f : lower;
		return std::clamp(komi, 0.5f, boardSize - 0.5f);
	}
}

#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include "consts.h"

// not "random": glibc declares a global random() function, which a namespace of the same name collides with.
namespace rnd {
	// one generator per thread, so self-play threads never share generator state.
	extern thread_local std::mt19937 gen;

	std::vector<int> select_indices(int range, int many);

	int pick0or1(float zeroProb);

	std::vector<float> sample_dirichlet(int k, float alpha);

	// self-play komi : baseKomi with probability 1 - randomProb, otherwise baseKomi + d for an integer d drawn uniformly
	// from [-range, range], restricted so the result stays at least 0.5. Keeps baseKomi's fractional part (x.5).
	float sample_komi(float baseKomi, float randomProb, int range);

	// neutral stones of a self-play game : none, or one stone. Every starting position up to rotation / flip is equally
	// likely (each symmetry class of points, plus "no neutral", has the same probability); within a class the point is
	// uniform, so all orientations appear.
	std::vector<Move> sample_neutrals();
}

#endif

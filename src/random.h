#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <numeric>
#include <algorithm>
#include <vector>

// not "random": glibc declares a global random() function, which a namespace of the same name collides with.
namespace rnd {
	// one generator per thread, so self-play threads never share generator state.
	extern thread_local std::mt19937 gen;

	std::vector<int> select_indices(int range, int many);

	int pick0or1(float zeroProb);

	std::vector<float> sample_dirichlet(int k, float alpha);
}

#endif

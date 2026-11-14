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

	// the board points grouped into their classes under the 8 rotations / flips, in a fixed order.
	const std::vector<std::vector<Move>>& neutral_point_classes();

	// neutral setups up to rotation / flip : one per point class, plus "no neutral" (the last index).
	int neutral_setup_count();

	// index of a setup of at most one neutral stone in [0, neutral_setup_count()); -1 if the point is off the board.
	int neutral_setup_index(const std::vector<Move>& neutrals);

	// neutral stones of a self-play game : none, or one stone. Every starting position up to rotation / flip is equally
	// likely (each symmetry class of points, plus "no neutral", has the same probability); within a class the point is
	// uniform, so all orientations appear.
	std::vector<Move> sample_neutrals();

	// komi around a setup's fair komi : uniform in [fairKomi - noise, fairKomi + noise] (cut below at 0.5), then rounded
	// to one of the two nearest x.5 values at random so that the mean stays that of the uniform draw. With probability
	// wideProb, wideNoise is used instead of noise : rare far-off komi so the value head stays sensible outside the
	// usual range (KataGo's komiBigStdevProb). Never below 0.5.
	float sample_fair_komi(float fairKomi, float noise, float wideProb = 0.0f, float wideNoise = 0.0f);
}

#endif

#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <numeric>
#include <algorithm>
#include <vector>

extern std::random_device rd;
extern std::mt19937 gen;

std::vector<int> select_indices(int range, int many);

#endif
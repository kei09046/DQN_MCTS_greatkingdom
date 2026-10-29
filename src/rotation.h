#ifndef ROTATION_H
#define ROTATION_H

#include <vector>
#include <array>
#include "neuralNet.h"

using InputMatrix = std::vector<float>;
using OutputMatrix = std::vector<float>;

template <typename T>
using Matrix = std::vector<T>;

std::vector<std::shared_ptr<TrainData>> generateDihedralTransformations(const TrainData& data);

#endif
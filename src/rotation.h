#ifndef ROTATION_H
#define ROTATION_H

#include <vector>
#include <array>
#include "neuralNet.h"

using InputMatrix = std::vector<float>;
using OutputMatrix = std::vector<float>;
using Matrix = std::vector<float>;

// InputMatrix inputRotate90(const InputMatrix mat);

// InputMatrix inputReflectHorizontal(const InputMatrix mat);

// std::vector<InputMatrix> generateTransformedInput(const InputMatrix mat);


// OutputMatrix outputRotate90(const OutputMatrix mat);

// OutputMatrix outputReflectHorizontal(const OutputMatrix mat);

// std::vector<OutputMatrix> generateTransformedOutput(const OutputMatrix mat);

std::vector<std::shared_ptr<TrainData>> generateDihedralTransformations(const TrainData& data);

// PolicyValueOutput rotateNNOutput(const PolicyValueOutput& original, const std::vector<std::pair<int, int>>& legal, int s, int N);

// std::vector<PolicyValueOutput> rotateAllNNOutputs(const PolicyValueOutput& original, const std::vector<Move>& legal, int N);

// std::pair<PolicyValueOutput, std::vector<Move>> rotateNNOutputandLegal(const PolicyValueOutput& original,
//                const std::vector<Move>& legal, int N, int s); 
#endif
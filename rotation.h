#ifndef ROTATION_H
#define ROTATION_H

#include <iostream>
#include <vector>
#include <array>
#include "neuralNet.h"

InputMatrix rotate90(const InputMatrix mat);

InputMatrix reflectHorizontal(const InputMatrix mat);

std::vector<InputMatrix> generateDihedralTransformations(const InputMatrix mat);


OutputMatrix rotate90(const OutputMatrix mat);

OutputMatrix reflectHorizontal(const OutputMatrix mat);

std::vector<OutputMatrix> generateDihedralTransformations(const OutputMatrix mat);

std::vector<TrainData*> generateDihedralTransformations(const TrainData& data);

inline std::pair<int,int> rot(int s, int r, int c, int N) {
    switch(s){
        case 0: return {r, c};           // identity
        case 1: return {c, N-1-r};       // rot90
        case 2: return {N-1-r, N-1-c};   // rot180
        case 3: return {N-1-c, r};       // rot270
        case 4: return {r, N-1-c};       // flipH
        case 5: return {N-1-r, c};       // flipV
        case 6: return {N-1-c, N-1-r};   // anti-diag
        case 7: return {c, r};           // diag
    }
    return {r, c};
}

PolicyValueOutput rotateNNOutput(const PolicyValueOutput& original, const std::vector<std::pair<int, int>>& legal, int s, int N);

std::vector<PolicyValueOutput> rotateAllNNOutputs(const PolicyValueOutput& original, const std::vector<std::pair<int,int>>& legal, int N);

std::pair<PolicyValueOutput, std::vector<std::pair<int,int>>> rotateNNOutputandLegal(const PolicyValueOutput& original,
               const std::vector<std::pair<int,int>>& legal, int N, int s); 
#endif
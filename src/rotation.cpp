#include "rotation.h"
#include <iostream>

namespace{
    template <typename T>
    Matrix<T> rotate90(const Matrix<T>& mat, int channel, bool passInc){
        Matrix<T> res;
        if(passInc)
            res.reserve((boardSize + 1) * channel);
        else
            res.reserve(boardSize * channel);

        int dnt = 0;
        for(int k = 0; k < channel; ++k){
            for (int i = 0; i < inputRow; ++i)
                for (int j = 0; j < inputCol; ++j)
                    res.push_back(mat[dnt + (inputRow - 1 - j) * inputCol + i]);
            
            dnt += boardSize;
            if(passInc)
                res.push_back(mat[dnt++]);
        }
        return res;
    }

    template <typename T>
    Matrix<T> reflectHorizontal(const Matrix<T>& mat, int channel, bool passInc){
        Matrix<T> res;
        if(passInc)
            res.reserve((boardSize + 1) * channel);
        else
            res.reserve(boardSize * channel);

        int dnt = 0;
        for(int k=0; k<channel; ++k){
            for (int i = 0; i < inputRow; ++i)
                for (int j = 0; j < inputCol; ++j)
                    res.push_back(mat[dnt + (inputRow - 1 - i) * inputCol + j]);

            dnt += inputSize;
            if(passInc)
                res.push_back(mat[dnt++]);
        }
        return res;
    }

    // available_moves is a flat list of board-position VALUES (not a position-indexed grid, unlike
    // transfer_table/board channels), so rotating it means remapping each value through the board's
    // coordinate transform, not permuting which array slot holds which value.
    // -1 (padding) and boardSize (PASS) are not board squares, so they pass through unchanged.
    inline int rotateIndex90(int v){
        if(v < 0 || v >= boardSize) return v;
        int r = v / colSize, c = v % colSize;
        // mirrors rotate90's res[i][j] = mat[N-1-j][i]: source (r,c) lands at target (c, N-1-r)
        return c * colSize + (colSize - 1 - r);
    }

    inline int reflectIndexHorizontal(int v){
        if(v < 0 || v >= boardSize) return v;
        int r = v / colSize, c = v % colSize;
        // mirrors reflectHorizontal's res[i][j] = mat[N-1-i][j]: source (r,c) lands at target (N-1-r, c)
        return (rowSize - 1 - r) * colSize + c;
    }

    template <typename T>
    Matrix<T> remapMoveList(const Matrix<T>& mat, int (*mapFn)(int)){
        Matrix<T> res;
        res.reserve(mat.size());
        for(const auto& v : mat)
            res.push_back(mapFn(v));
        return res;
    }

    // Same 8-transform composition/order as generateTransformed below, so the result lines up
    // index-for-index with rotatedStates/rotatedTransfer/rotatedMoves/rotatedMap.
    template <typename T>
    std::vector<Matrix<T>> generateTransformedMoveList(const Matrix<T>& mat){
        std::vector<Matrix<T>> transforms;
        transforms.reserve(8);

        transforms.push_back(mat);

        Matrix<T> rot90 = remapMoveList(mat, rotateIndex90);
        Matrix<T> rot180 = remapMoveList(rot90, rotateIndex90);
        Matrix<T> rot270 = remapMoveList(rot180, rotateIndex90);

        transforms.push_back(rot90);
        transforms.push_back(rot180);
        transforms.push_back(rot270);

        Matrix<T> reflH = remapMoveList(mat, reflectIndexHorizontal);
        Matrix<T> ref_rot90 = remapMoveList(reflH, rotateIndex90);
        Matrix<T> ref_rot180 = remapMoveList(ref_rot90, rotateIndex90);
        Matrix<T> ref_rot270 = remapMoveList(ref_rot180, rotateIndex90);

        transforms.push_back(reflH);
        transforms.push_back(ref_rot90);
        transforms.push_back(ref_rot180);
        transforms.push_back(ref_rot270);

        return transforms;
    }

    template <typename T>
    std::vector<Matrix<T>> generateTransformed(const Matrix<T> mat, int channel, bool passInc){
        std::vector<Matrix<T>> transforms;
        transforms.reserve(8);
        
        // Original Matrix<T>
        transforms.push_back(mat);

        // Rotations
        Matrix<T> rot90 = rotate90(mat, channel, passInc);
        Matrix<T> rot180 = rotate90(rot90, channel, passInc);
        Matrix<T> rot270 = rotate90(rot180, channel, passInc);

        transforms.push_back(rot90);
        transforms.push_back(rot180);
        transforms.push_back(rot270);

        // Reflections
        Matrix<T> reflH = reflectHorizontal(mat, channel, passInc);
        Matrix<T> ref_rot90 = rotate90(reflH, channel, passInc);
        Matrix<T> ref_rot180 = rotate90(ref_rot90, channel, passInc);
        Matrix<T> ref_rot270 = rotate90(ref_rot180, channel, passInc);

        transforms.push_back(reflH);
        transforms.push_back(ref_rot90);
        transforms.push_back(ref_rot180);
        transforms.push_back(ref_rot270);

        return transforms;
    }

    inline Move rot(int s, int r, int c, int N) {
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
}

std::vector<std::shared_ptr<TrainData>> generateDihedralTransformations(const TrainData& data) {
    std::vector<std::shared_ptr<TrainData>> transformed_data;
    transformed_data.reserve(8);
    
    auto& states = std::get<0>(data);

    // NN input rotation
    auto rotatedStates = generateTransformed(states, globalConfig.inputChannel, false);

    auto rotatedMoves = generateTransformed(std::get<1>(data), 1, true); // NN output rotation
    auto value = std::get<2>(data);
    auto scoreDiff = std::get<3>(data);
    auto rotatedMap = generateTransformed(std::get<4>(data), 1, false); // score map/capture map rotation
    auto type = std::get<5>(data);

    for(int i=0; i<8; ++i){
        transformed_data.push_back(std::make_shared<TrainData>(rotatedStates[i],
             rotatedMoves[i], value, scoreDiff, rotatedMap[i], type));
    }

    return transformed_data;
}

// PolicyValueOutput rotateNNOutput(const PolicyValueOutput& original,
//                                  const std::vector<Move>& legal,
//                                  int s, int N) 
// {
//     const auto& policy = std::get<0>(original);
//     const auto value = std::get<1>(original);
//     const auto score = std::get<2>(original);
//     const auto score_dist = std::get<3>(original);
//     int L = legal.size();

//     // Compute rotated legal positions
//     std::vector<Move> rotated(L);
//     for (int i = 0; i < L; ++i)
//         rotated[i] = rot(s, legal[i].first, legal[i].second, N);

//     // Create index array
//     std::vector<int> idx(L);
//     for (int i = 0; i < L; ++i) idx[i] = i;

//     // Sort indices based on rotated legal positions
//     std::sort(idx.begin(), idx.end(),
//               [&](int a, int b){ return rotated[a] < rotated[b]; });

//     // Build new policy vector in sorted order
//     std::vector<float> new_policy(L);
//     for (int i = 0; i < L; ++i)
//         new_policy[i] = policy[idx[i]];

//     return {new_policy, value, score, score_dist};
// }

// Returns rotated PolicyValueOutput AND rotated legal moves
// std::pair<PolicyValueOutput, std::vector<Move>> rotateNNOutputandLegal(const PolicyValueOutput& original,
//                const std::vector<Move>& legal, int N, int s) 
// {
//     const auto& policy = std::get<0>(original);
//     const auto value = std::get<1>(original);
//     const auto score = std::get<2>(original);
//     const auto score_diff = std::get<3>(original);
//     int L = legal.size();

//     // Compute rotated legal positions
//     std::vector<Move> rotated_legal(L);
//     for (int i = 0; i < L; ++i)
//         rotated_legal[i] = rot(s, legal[i].first, legal[i].second, N);

//     // Create index array
//     std::vector<int> idx(L);
//     for (int i = 0; i < L; ++i) idx[i] = i;

//     // Sort indices based on rotated legal positions
//     std::sort(idx.begin(), idx.end(),
//               [&](int a, int b){ return rotated_legal[a] < rotated_legal[b]; });

//     // Build new policy vector in sorted order
//     std::vector<float> new_policy(L);
//     for (int i = 0; i < L; ++i)
//         new_policy[i] = policy[idx[i]];

//     // Reorder legal in the same way
//     std::vector<Move> new_legal(L);
//     for (int i = 0; i < L; ++i)
//         new_legal[i] = rotated_legal[idx[i]];

//     return {{new_policy, value, score, score_diff}, new_legal};
// }


// std::vector<PolicyValueOutput> rotateAllNNOutputs(
//     const PolicyValueOutput& original,
//     const std::vector<Move>& legal,
//     int N)
// {
//     std::vector<PolicyValueOutput> outputs;
//     outputs.reserve(8);

//     for (int s = 0; s < 8; ++s) {
//         outputs.push_back(rotateNNOutput(original, legal, s, N));
//     }

//     return outputs;
// }

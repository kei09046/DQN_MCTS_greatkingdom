#include "rotation.h"
#include <iostream>

namespace{
    Matrix rotate90(const Matrix& mat, int channel, bool passInc){
        Matrix res;
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

    Matrix reflectHorizontal(const Matrix& mat, int channel, bool passInc){
        Matrix res;
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

    std::vector<Matrix> generateTransformed(const Matrix mat, int channel, bool passInc){
        std::vector<Matrix> transforms;
        transforms.reserve(8);
        
        // Original Matrix
        transforms.push_back(mat);

        // Rotations
        Matrix rot90 = rotate90(mat, channel, passInc);
        Matrix rot180 = rotate90(rot90, channel, passInc);
        Matrix rot270 = rotate90(rot180, channel, passInc);

        transforms.push_back(rot90);
        transforms.push_back(rot180);
        transforms.push_back(rot270);

        // Reflections
        Matrix reflH = reflectHorizontal(mat, channel, passInc);
        Matrix ref_rot90 = rotate90(reflH, channel, passInc);
        Matrix ref_rot180 = rotate90(ref_rot90, channel, passInc);
        Matrix ref_rot270 = rotate90(ref_rot180, channel, passInc);

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


// InputMatrix inputRotate90(const InputMatrix mat) {
//     assert(mat.size() == inputSize * globalConfig.inputChannel);

//     InputMatrix res(inputSize * globalConfig.inputChannel);
//     int cnt = 0, dnt = 0;

//     for(int k = 0; k < globalConfig.inputChannel; ++k){
//         for (int i = 0; i < inputRow; ++i)
//             for (int j = 0; j < inputCol; ++j)
//                 res[cnt++] = mat[dnt + (inputRow - 1 - j) * inputCol + i];
        
//         dnt += inputSize;
//     }
//     return res;
// }

// InputMatrix inputReflectHorizontal(const InputMatrix mat) {
//     assert(mat.size() == inputSize * globalConfig.inputChannel);

//     InputMatrix res(inputSize * globalConfig.inputChannel);
//     int cnt = 0, dnt = 0;

//     for(int k=0; k<globalConfig.inputChannel; ++k){
//         for (int i = 0; i < inputRow; ++i)
//             for (int j = 0; j < inputCol; ++j)
//                 res[cnt++] = mat[dnt + (inputRow - 1 - i) * inputCol + j];

//         dnt += inputSize;
//     }
//     return res;
// }

// std::vector<InputMatrix> generateTransformedInput(const InputMatrix mat) {
//     std::vector<InputMatrix> transforms;
//     transforms.reserve(8);
    
//     // Original InputMatrix
//     transforms.push_back(mat);

//     // Rotations
//     InputMatrix rot90 = inputRotate90(mat);
//     InputMatrix rot180 = inputRotate90(rot90);
//     InputMatrix rot270 = inputRotate90(rot180);

//     transforms.push_back(rot90);
//     transforms.push_back(rot180);
//     transforms.push_back(rot270);

//     // Reflections
//     InputMatrix reflH = inputReflectHorizontal(mat);
//     InputMatrix ref_rot90 = inputRotate90(reflH);
//     InputMatrix ref_rot180 = inputRotate90(ref_rot90);
//     InputMatrix ref_rot270 = inputRotate90(ref_rot180);

//     transforms.push_back(reflH);
//     transforms.push_back(ref_rot90);
//     transforms.push_back(ref_rot180);
//     transforms.push_back(ref_rot270);

//     return transforms;
// }


// OutputMatrix outputRotate90(const OutputMatrix mat) {
//     assert(mat.size() == outputSize);

//     OutputMatrix res(outputSize);
//     int cnt = 0;

//     for (int i = 0; i < outputRow; ++i)
//         for (int j = 0; j < outputCol; ++j)
//             res[cnt++] = mat[(outputRow - 1 - j) * outputCol + i];

//     res[cnt] = mat[cnt];
//     return res;
// }

// OutputMatrix outputReflectHorizontal(const OutputMatrix mat) {
//     assert(mat.size() == outputSize);

//     OutputMatrix res(outputSize);
//     int cnt = 0;

//     for (int i = 0; i < outputRow; ++i)
//         for (int j = 0; j < outputCol; ++j)
//             res[cnt++] = mat[(outputRow - 1 - i) * outputCol + j];

//     res[cnt] = mat[cnt];
//     return res;
// }

// std::vector<OutputMatrix> generateTransformedOutput(const OutputMatrix mat) {
//     std::vector<OutputMatrix> transforms;
    
//     // Original OutputMatrix
//     transforms.push_back(mat);

//     // Rotations
//     OutputMatrix rot90 = outputRotate90(mat);
//     OutputMatrix rot180 = outputRotate90(rot90);
//     OutputMatrix rot270 = outputRotate90(rot180);

//     transforms.push_back(rot90);
//     transforms.push_back(rot180);
//     transforms.push_back(rot270);

//     // Reflections
//     OutputMatrix reflH = outputReflectHorizontal(mat);
//     OutputMatrix ref_rot90 = outputRotate90(reflH);
//     OutputMatrix ref_rot180 = outputRotate90(ref_rot90);
//     OutputMatrix ref_rot270 = outputRotate90(ref_rot180);

//     transforms.push_back(reflH);
//     transforms.push_back(ref_rot90);
//     transforms.push_back(ref_rot180);
//     transforms.push_back(ref_rot270);

//     return transforms;
// }

std::vector<std::shared_ptr<TrainData>> generateDihedralTransformations(const TrainData& data) {
    std::vector<std::shared_ptr<TrainData>> transformed_data;
    transformed_data.reserve(8);
    
    auto rotatedStates = generateTransformed(std::get<0>(data), globalConfig.inputChannel, false); // NN input rotation
    auto rotatedMoves = generateTransformed(std::get<1>(data), 1, true); // NN output rotation
    auto value = std::get<2>(data);
    auto scoreDiff = std::get<3>(data);
    auto rotatedMap = generateTransformed(std::get<4>(data), 1, false); // score map/capture map rotation
    auto type = std::get<5>(data);

    for(int i=0; i<8; ++i){
        transformed_data.push_back(std::make_shared<TrainData>(rotatedStates[i], rotatedMoves[i], value, scoreDiff, rotatedMap[i], type));
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

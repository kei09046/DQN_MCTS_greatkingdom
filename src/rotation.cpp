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
    Matrix<T> reflect_horizontal(const Matrix<T>& mat, int channel, bool passInc){
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
    inline int rotate_index90(int v){
        if(v < 0 || v >= boardSize) return v;
        int r = v / colSize, c = v % colSize;
        // mirrors rotate90's res[i][j] = mat[N-1-j][i]: source (r,c) lands at target (c, N-1-r)
        return c * colSize + (colSize - 1 - r);
    }

    inline int reflect_index_horizontal(int v){
        if(v < 0 || v >= boardSize) return v;
        int r = v / colSize, c = v % colSize;
        // mirrors reflectHorizontal's res[i][j] = mat[N-1-i][j]: source (r,c) lands at target (N-1-r, c)
        return (rowSize - 1 - r) * colSize + c;
    }

    template <typename T>
    Matrix<T> remap_move_list(const Matrix<T>& mat, int (*mapFn)(int)){
        Matrix<T> res;
        res.reserve(mat.size());
        for(const auto& v : mat)
            res.push_back(mapFn(v));
        return res;
    }

    // Same 8-transform composition/order as generateTransformed below, so the result lines up
    // index-for-index with rotatedStates/rotatedTransfer/rotatedMoves/rotatedMap.
    template <typename T>
    std::vector<Matrix<T>> generate_transformed_move_list(const Matrix<T>& mat){
        std::vector<Matrix<T>> transforms;
        transforms.reserve(8);

        transforms.push_back(mat);

        Matrix<T> rot90 = remap_move_list(mat, rotate_index90);
        Matrix<T> rot180 = remap_move_list(rot90, rotate_index90);
        Matrix<T> rot270 = remap_move_list(rot180, rotate_index90);

        transforms.push_back(rot90);
        transforms.push_back(rot180);
        transforms.push_back(rot270);

        Matrix<T> reflH = remap_move_list(mat, reflect_index_horizontal);
        Matrix<T> refRot90 = remap_move_list(reflH, rotate_index90);
        Matrix<T> refRot180 = remap_move_list(refRot90, rotate_index90);
        Matrix<T> refRot270 = remap_move_list(refRot180, rotate_index90);

        transforms.push_back(reflH);
        transforms.push_back(refRot90);
        transforms.push_back(refRot180);
        transforms.push_back(refRot270);

        return transforms;
    }

    template <typename T>
    std::vector<Matrix<T>> generate_transformed(const Matrix<T> mat, int channel, bool passInc){
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
        Matrix<T> reflH = reflect_horizontal(mat, channel, passInc);
        Matrix<T> refRot90 = rotate90(reflH, channel, passInc);
        Matrix<T> refRot180 = rotate90(refRot90, channel, passInc);
        Matrix<T> refRot270 = rotate90(refRot180, channel, passInc);

        transforms.push_back(reflH);
        transforms.push_back(refRot90);
        transforms.push_back(refRot180);
        transforms.push_back(refRot270);

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

std::vector<std::shared_ptr<TrainData>> generate_dihedral_transformations(const TrainData& data) {
    std::vector<std::shared_ptr<TrainData>> transformedData;
    transformedData.reserve(8);
    
    auto& states = std::get<0>(data);

    // NN input rotation
    auto rotatedStates = generate_transformed(states, globalConfig.inputChannel, false);

    auto rotatedMoves = generate_transformed(std::get<1>(data), 1, true); // NN output rotation
    auto value = std::get<2>(data);
    auto scoreDiff = std::get<3>(data);
    auto rotatedMap = generate_transformed(std::get<4>(data), 1, false); // score map/capture map rotation
    auto type = std::get<5>(data);

    for(int i=0; i<8; ++i){
        transformedData.push_back(std::make_shared<TrainData>(rotatedStates[i],
             rotatedMoves[i], value, scoreDiff, rotatedMap[i], type, std::get<6>(data)));
    }

    return transformedData;
}
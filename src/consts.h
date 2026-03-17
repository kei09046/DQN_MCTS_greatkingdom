#ifndef CONSTS_H
#define CONSTS_H

#include <utility>
#include <string>
#include <cstdint>
#include <vector>
#include <array>
#include "config.h"


// for hash
using HashValue = uint64_t;
using PolicyValueOutput = std::tuple<std::vector<float>, std::vector<float>, float, std::vector<float>>; // policy, expected result, score diff, score distribution

//mcts constants
using Move = std::pair<uint8_t, uint8_t>;
using MoveData = std::tuple<Move, std::vector<float> >; // move + move probability

using Color = uint8_t;
using Wintype = uint8_t;
using TrainData = std::tuple<std::vector<float>, std::vector<float>, Wintype, float>; // board, moveprob, outcome, score diff


constexpr int rowSize = 9;
constexpr int colSize = 9;
constexpr int boardSize = rowSize * colSize;
constexpr int outputSize = boardSize + 1; // board place + pass
constexpr int inputRow = rowSize;
constexpr int inputCol = colSize;
constexpr int outputRow = rowSize;
constexpr int outputCol = colSize;
constexpr int inputSize = inputRow * inputCol;

constexpr Color BLACK = 0;
constexpr Color WHITE = 1;
constexpr Color NEUTRAL = 2;
constexpr Color EMPTY = 4;

constexpr Wintype CAPTURE = 0;
constexpr Wintype SCORE = 1;
constexpr Wintype RESIGN = 2;
constexpr Wintype NONE = 3;

constexpr uint8_t ADJTOBLACK = 8U;
constexpr uint8_t ADJTOWHITE = 16U;

constexpr Move PASSMOVE = {rowSize, 0};
constexpr Move RESIGNMOVE = {255, 255};
#endif
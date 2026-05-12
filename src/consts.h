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
using PolicyValueOutput = std::tuple<std::vector<float>, float, float, std::vector<float>, std::vector<float>>; // policy, winP, score diff, score map, capture map

//mcts constants
using Move = std::pair<uint8_t, uint8_t>;
using MoveData = std::tuple<Move, std::vector<float> >; // move + move probability

using Color = uint8_t;
using Wintype = uint8_t;
using TrainData = std::tuple<std::vector<float>, std::vector<float>, float, float, std::vector<float>>; // board, moveprob, outcome, score diff, (score map/capture map)


constexpr int rowSize = 9;
constexpr int colSize = 9;
constexpr int boardSize = rowSize * colSize;
constexpr int outputSize = boardSize + 1; // board place + pass
constexpr int inputRow = rowSize;
constexpr int inputCol = colSize;
constexpr int outputRow = rowSize;
constexpr int outputCol = colSize;
constexpr int inputSize = inputRow * inputCol;

constexpr int PLAYOUT = 0;
constexpr int TIMEOUT = 1;

constexpr Color BLACK = 0U;
constexpr Color WHITE = 1U;
constexpr Color NEUTRAL = 2U;
constexpr Color EMPTY = 4U;

constexpr Wintype CAPTURE = 0U;
constexpr Wintype SCORE = 1U;
constexpr Wintype RESIGN = 2U;
constexpr Wintype NONE = 3U;

constexpr uint8_t ADJTOBLACK = 8U;
constexpr uint8_t ADJTOWHITE = 16U;
constexpr uint8_t POTSCORE = 32U;

constexpr Move PASSMOVE = {rowSize, 0};
constexpr Move RESIGNMOVE = {255, 255};
#endif
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
using PolicyValueOutput = std::pair<std::vector<float>, float>;

//mcts constants
using Move = std::pair<uint8_t, uint8_t>;
using MoveData = std::tuple<Move, std::vector<float> >; // move + move possibility

using InputMatrix = std::vector<float>;
using OutputMatrix = std::vector<float>;
using delete_flag = uint8_t; // decides whether data gets deleted during buffer replacement or training
using TrainData = std::tuple<InputMatrix, OutputMatrix, float, delete_flag>;
using color = uint8_t;


constexpr int rowSize = 9;
constexpr int colSize = 9;
constexpr int boardSize = rowSize * colSize;
constexpr int outputSize = boardSize + 1; // board place + pass
constexpr int inputRow = rowSize;
constexpr int inputCol = colSize;
constexpr int outputRow = rowSize;
constexpr int outputCol = colSize;
constexpr int inputSize = inputRow * inputCol;

constexpr color BLACK = 0U;
constexpr color WHITE = 1U;
constexpr color NEUTRAL = 2U;
constexpr color EMPTY = 4U;
constexpr uint8_t ADJTOBLACK = 8U;
constexpr uint8_t ADJTOWHITE = 16U;

constexpr Move PASSMOVE = {rowSize, 0};
constexpr Move RESIGNMOVE = {255, 255};
#endif
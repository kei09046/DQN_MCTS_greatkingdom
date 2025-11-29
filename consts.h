#ifndef CONSTS_H
#define CONSTS_H

#include <utility>
#include <string>
#include <cstdint>
#include <vector>

using u_int = unsigned int;

// for hash
using HashValue = uint64_t;
using PolicyValueOutput = std::pair<std::vector<float>, float>;

// rating constants
constexpr float init_K = 32.0f;
constexpr float init_rating = 0.0f;

// gamerules constants
using color = uint8_t;
constexpr color BLACK = 0U;
constexpr color WHITE = 1U;
constexpr color NEUTRAL = 2U;
constexpr color EMPTY = 4U;
constexpr uint8_t ADJTOBLACK = 8U;
constexpr uint8_t ADJTOWHITE = 16U;

constexpr u_int rowSize = 7;
constexpr u_int colSize = 7;
constexpr int boardSize = rowSize * colSize;
constexpr float komi = 2.5f;
constexpr std::pair<int, int> neutral = {3, 3};
constexpr char dr[4] = {-1, 0, 1, 0};
constexpr char dc[4] = {0, 1, 0, -1};
// constexpr std::vector<std::pair<int, int>> adjCells;

//policyvalue constants
constexpr u_int batchSize = 512;
constexpr u_int inputRow = rowSize;
constexpr u_int inputCol = colSize;
constexpr u_int outputRow = rowSize;
constexpr u_int outputCol = colSize;
constexpr u_int inputSize = inputRow * inputCol;
constexpr u_int inputDepth = 6;
constexpr u_int outputSize = boardSize + 1; // board place + pass
constexpr float cPuct = 2.0f;

//evalcache constants
constexpr u_int tableSize = 1 << 20;
constexpr u_int mutexPoolSize = 1 << 10; // shardCount * capPerShard -> maximum cache size

//train constants
constexpr u_int n_playout = 400;
constexpr u_int play_batch_size = 1;
constexpr u_int epochs = 5;
constexpr u_int check_freq = 100;
constexpr u_int save_freq = 100;
constexpr u_int capacity = 10000;
constexpr float thres = 0.1f;
const std::string model_path = "./models/";

#endif
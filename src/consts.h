#ifndef CONSTS_H
#define CONSTS_H

#include <utility>
#include <string>
#include <cstdint>
#include <vector>
#include <array>
#include <optional>
#include <iostream>
#include "config.h"


// for hash
using HashValue = uint64_t;
//mcts constants
using Move = std::pair<uint8_t, uint8_t>;
// move + move probability + forcedState + onlyMove? + root value after search (see Node::selectMoveWithStat)
using MoveData = std::tuple<Move, std::vector<float>, int, bool, std::optional<float>>;

// for training heads
using Trainhead = uint8_t;

using Color = uint8_t;


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

// board status info.
constexpr Color BLACK = 1U;
constexpr Color WHITE = 2U;
constexpr Color NEUTRAL = 4U;
constexpr Color EMPTY = 8U;
constexpr uint8_t ADJTOBLACK = 16U;
constexpr uint8_t ADJTOWHITE = 32U;
constexpr uint8_t BSCORE = 64U;
constexpr uint8_t WSCORE = 128U;
constexpr uint8_t BOARDMASK = EMPTY | BLACK | NEUTRAL | WHITE;
constexpr uint8_t SCOREMASK = BSCORE | WSCORE;

constexpr Trainhead POLICYHEAD = 1U;
constexpr Trainhead VALUEHEAD = 2U;
constexpr Trainhead SCOREHEAD = 4U;
constexpr Trainhead OCCUPYHEAD = 8U;
constexpr Trainhead CMAPHEAD = 16U;
constexpr Trainhead ALL = 255U;


constexpr Move PASSMOVE = {rowSize, 0};
constexpr Move RESIGNMOVE = {255, 255};

inline void print_move(const Move& m){
    std::cerr << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;
}

// in-bound neighbors of a square, precomputed at compile time.
// usage : for(auto [nr, nc] : neighbors4(r, c)) { ... }   (break / continue / return work as usual)
template <int N>
struct Neighbors{
    std::array<Move, N> cells{};
    uint8_t count = 0;
    constexpr const Move* begin() const { return cells.data(); }
    constexpr const Move* end() const { return cells.data() + count; }
};

// builds neighbor table from direction offsets. Out-of-bound neighbors are skipped, order of dr/dc is kept.
template <int N>
constexpr std::array<Neighbors<N>, boardSize> make_neighbor_table(const int (&dr)[N], const int (&dc)[N]){
    std::array<Neighbors<N>, boardSize> table{};
    for(int r=0; r<rowSize; ++r)
        for(int c=0; c<colSize; ++c){
            auto& nb = table[r * colSize + c];
            for(int i=0; i<N; ++i){
                const int nr = r + dr[i], nc = c + dc[i];
                if(nr >= 0 && nr < rowSize && nc >= 0 && nc < colSize)
                    nb.cells[nb.count++] = {static_cast<uint8_t>(nr), static_cast<uint8_t>(nc)};
            }
        }
    return table;
}

// order : up, right, down, left
inline constexpr int DR4[4] = {-1, 0, 1, 0};
inline constexpr int DC4[4] = {0, 1, 0, -1};
inline constexpr auto NEIGHBORS4 = make_neighbor_table(DR4, DC4);

// order : clockwise from up-left (up-left, up, up-right, right, down-right, down, down-left, left)
inline constexpr int DR8[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
inline constexpr int DC8[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
inline constexpr auto NEIGHBORS8 = make_neighbor_table(DR8, DC8);

// r, c must be on the board (not PASSMOVE / RESIGNMOVE).
inline constexpr const Neighbors<4>& neighbors4(int r, int c){
    return NEIGHBORS4[r * colSize + c];
}

inline constexpr const Neighbors<4>& neighbors4(const Move& m){
    return neighbors4(m.first, m.second);
}

inline constexpr const Neighbors<8>& neighbors8(int r, int c){
    return NEIGHBORS8[r * colSize + c];
}

inline constexpr const Neighbors<8>& neighbors8(const Move& m){
    return neighbors8(m.first, m.second);
}
#endif
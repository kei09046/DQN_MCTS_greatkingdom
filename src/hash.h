#ifndef HASH_H
#define HASH_H

#include <vector>
#include <random>
#include <cstdint>
#include "consts.h"
#include "gamerules.h"

class Hash{
private:
    HashValue zobristTable[rowSize][colSize][4]; // 4 for BLACK, WHITE, NEUTRAL, EMPTY
    HashValue zobristToPlay; // to encode whose turn it is

    static inline int color_to_index(Color c);

public:
    Hash();

    HashValue base_hash() const;
    HashValue compute_hash(const Game& game) const;
    HashValue compute_hash_after_move(const Game& game, const std::pair<int, int>& move, const HashValue prevHash) const;
};

#endif
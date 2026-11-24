#ifndef HASH_H
#define HASH_H

#include <vector>
#include <random>
#include "consts.h"
#include "gamerules.h"

class Hash{
private:
    HashValue zobristTable[rowSize][colSize][4]; // 4 for BLACK, WHITE, NEUTRAL, EMPTY
    HashValue zobristToPlay; // to encode whose turn it is
    HashValue komiHash[boardSize * 2]; // komi can be -boardSize ~ boardSize.

    static inline int color_to_index(Color c);

public:
    Hash();

    HashValue compute_hash(const Game& game) const;
    HashValue compute_hash_after_move(const Game& game, const std::pair<int, int>& move, const HashValue prevHash) const;
};

#endif
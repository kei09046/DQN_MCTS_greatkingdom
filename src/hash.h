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

    static inline int colorToIndex(Color c);

public:
    Hash();

    HashValue baseHash() const;
    HashValue computeHash(const Game& game) const;
    HashValue computeHashAfterMove(const Game& game, const std::pair<int, int>& move, const HashValue prevHash) const;
};

#endif
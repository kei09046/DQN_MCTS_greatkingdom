#include "hash.h"

size_t Hash::colorToIndex(color c){
    switch(c){
        case BLACK:
            return 0;
        case WHITE:
            return 1;
        case NEUTRAL:
            return 2;
        default:
            return 3;
    }
}

Hash::Hash() {
        std::mt19937_64 rng(42); // fixed seed for reproducibility
        std::uniform_int_distribution<HashValue> dist;
        for (int r = 0; r < rowSize; ++r)
            for (int c = 0; c < colSize; ++c)
                for (int color = 0; color < 3; ++color)
                    zobristTable[r][c][color] = dist(rng);
        zobristToPlay = dist(rng);
    }

HashValue Hash::baseHash() const{
    return zobristTable[neutral.first][neutral.second][2];
}

HashValue Hash::computeHashAfterMove(const Game& old_game, const std::pair<int, int>& move, const HashValue prevHash) const{
    HashValue h = prevHash;
    h ^= zobristToPlay;
    if(move.first != rowSize){
        h ^= zobristTable[move.first][move.second][colorToIndex(old_game.getTurn())];
    }

    return h;
}
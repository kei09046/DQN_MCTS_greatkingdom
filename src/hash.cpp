#include "hash.h"

int Hash::color_to_index(Color c){
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
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<HashValue> dist;

    // for empty square, zobrist table has 0 value.
    for (int r = 0; r < rowSize; ++r)
        for (int c = 0; c < colSize; ++c)
            for (int color = 0; color < 3; ++color){
                zobristTable[r][c][color] = dist(rng);
                //std::cerr << color << " " << r << " " << c << " " << zobristTable[r][c][color] << std::endl;
            }
    zobristToPlay = dist(rng);
    //std::cerr << zobristToPlay << std::endl;
}

HashValue Hash::base_hash() const{
    static HashValue h = 0;
    if(h == 0){
        for(const auto& neutral : globalConfig.neutrals)
            h ^= zobristTable[neutral.first][neutral.second][2];
    }
    return h;
}

HashValue Hash::compute_hash(const Game& game) const{
    HashValue h = 0;

    if(game.get_turn() == WHITE)
        h ^= zobristToPlay;
    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            h ^= zobristTable[i][j][color_to_index(game.get_board({i, j}))];
        }
    }

    return h;
}

HashValue Hash::compute_hash_after_move(const Game& oldGame, const std::pair<int, int>& move, const HashValue prevHash) const{
    HashValue h = prevHash;
    h ^= zobristToPlay;
    if(move.first != rowSize){
        h ^= zobristTable[move.first][move.second][color_to_index(oldGame.get_turn())];
    }

    return h;
}
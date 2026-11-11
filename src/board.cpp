#include "board.h"

Board::Board(){
    emptyBB.set();
}

Board::Board(const std::vector<Move>& neutrals) : Board(){
    for(const auto& n : neutrals){
        const int idx = to_idx(n);
        emptyBB.reset(idx);
        neutralBB.set(idx);
    }
    assert(consistent());
}

void Board::place_stone(int idx, Color c){
    assert(emptyBB[idx]);
    const int ci = color_idx(c);
    emptyBB.reset(idx);
    stoneBB[ci].set(idx);
    for(auto [nr, nc] : neighbors4(idx / colSize, idx % colSize))
        adjBB[ci].set(to_idx(nr, nc));
    assert(consistent());
}

void Board::mark_score(const BitBoard& squares, Color c){
    assert((squares & ~emptyBB).none());
    emptyBB &= ~squares;
    scoreBB[color_idx(c)] |= squares;
    assert(consistent());
}

void Board::mark_score(int idx, Color c){
    assert(emptyBB[idx]);
    emptyBB.reset(idx);
    scoreBB[color_idx(c)].set(idx);
    assert(consistent());
}

Color Board::at(int idx) const{
    if(stoneBB[0][idx]) return BLACK;
    if(stoneBB[1][idx]) return WHITE;
    if(neutralBB[idx]) return NEUTRAL;
    if(emptyBB[idx]) return EMPTY;
    return 0U; // territory
}

Color Board::score_at(int idx) const{
    if(scoreBB[0][idx]) return BSCORE;
    if(scoreBB[1][idx]) return WSCORE;
    return 0U;
}

bool Board::consistent() const{
    const BitBoard states[] = {stoneBB[0], stoneBB[1], scoreBB[0], scoreBB[1], neutralBB, emptyBB};
    BitBoard seen;
    for(const auto& s : states){
        if((seen & s).any())   // a square in two states
            return false;
        seen |= s;
    }
    return seen.all();         // every square in some state
}

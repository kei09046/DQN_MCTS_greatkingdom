#ifndef BOARD_H
#define BOARD_H

#include "consts.h"

#include <bitset>
#include <cassert>

using BitBoard = std::bitset<boardSize>;

inline constexpr int to_idx(int r, int c){ return r * colSize + c; }
inline constexpr int to_idx(const Move& m){ return to_idx(m.first, m.second); }

// Board state stored as bitboards. Bit index = r * colSize + c.
// Every square is in exactly one of : black stone, white stone, neutral, black score, white score, empty.
// All mutation goes through place_stone / mark_score so that these bitboards (and adjacency) stay consistent.
// Squares only ever change empty -> stone or empty -> score, so nothing has to be cleared.
class Board{
public:
    Board();
    explicit Board(const std::vector<Move>& neutrals);

    ///////////////////////////// mutators /////////////////////////////
    // idx must be empty.
    void place_stone(int idx, Color c);
    inline void place_stone(const Move& m, Color c){ place_stone(to_idx(m), c); }

    // marks squares as territory of c. Every square must be empty.
    void mark_score(const BitBoard& squares, Color c);
    void mark_score(int idx, Color c);

    ////////////////////////// bitboard queries //////////////////////////
    inline const BitBoard& stones(Color c) const { return stoneBB[color_idx(c)]; }
    inline const BitBoard& neutral() const { return neutralBB; }
    inline const BitBoard& empty() const { return emptyBB; }
    inline const BitBoard& score(Color c) const { return scoreBB[color_idx(c)]; }
    // squares orthogonally adjacent to a stone of color c (replaces ADJTOBLACK / ADJTOWHITE).
    // Includes non-empty squares; mask with empty() if needed.
    inline const BitBoard& adjacent(Color c) const { return adjBB[color_idx(c)]; }
    // stones of color c + neutrals : what counts as 'own' for territory building.
    inline BitBoard own(Color c) const { return stoneBB[color_idx(c)] | neutralBB; }

    ///////////////////////// per-square queries /////////////////////////
    inline bool is_empty(int idx) const { return emptyBB[idx]; }
    inline bool is_empty(const Move& m) const { return is_empty(to_idx(m)); }

    // BLACK / WHITE / NEUTRAL / EMPTY, or 0 for territory. Same as old (board & BOARDMASK).
    Color at(int idx) const;
    inline Color at(const Move& m) const { return at(to_idx(m)); }

    // BSCORE / WSCORE, or 0. Same as old (board & SCOREMASK).
    Color score_at(int idx) const;
    inline Color score_at(const Move& m) const { return score_at(to_idx(m)); }

    // true if every square is in exactly one state. For assert() in debug builds.
    bool consistent() const;

private:
    BitBoard stoneBB[2];  // [0] black, [1] white
    BitBoard scoreBB[2];  // [0] black, [1] white
    BitBoard adjBB[2];    // [0] adjacent to black, [1] adjacent to white
    BitBoard neutralBB;
    BitBoard emptyBB;     // kept separately (not derived) so queries need no computation.

    static inline int color_idx(Color c){
        assert(c == BLACK || c == WHITE);
        return (c == BLACK) ? 0 : 1;
    }
};

#endif

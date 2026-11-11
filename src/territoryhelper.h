#ifndef TERRITORYHELPER
#define TERRITORYHELPER

#include "consts.h"
#include "board.h"

#include <bitset>

// structure needed for effective score computation
using SegId = uint8_t;
using GroupId = uint8_t;

struct SegInfo{
    std::bitset<boardSize> connectedSeg;    // bit t set <-> segment t is adjacent. SegIds are always < boardSize.
    bool adjOpp;
    std::bitset<4> adjEdge;
};

// segments of one color on one position. Built by update_segment_table(), only used inside update_topology().
// segMap[i] : segment of empty square i (BASESEGID for filled squares and empty regions not adjacent to any candidate).
// segInfos[s] : info of segment s. Number of segments <= number of empty squares, so ids < boardSize.
struct SegTable{
    std::array<SegInfo, boardSize> segInfos{};
    std::array<SegId, boardSize> segMap;
};

// helper class which computes score gaining moves on given position.
// Does not own the board : the board is passed on every call (Game owns it), so copying Game stays safe.
// Functions taking non-const Board& mark territory on it.
class TerritoryHelper{
public:
    TerritoryHelper();

    // Both must be called after the stone is placed on board (color is read from board), and also for PASSMOVE.
    // used to compute score without calling 'update_topology()' first. Slower.
    void update_move(Board& board, const Move& m);

    // fast version using groupIds from update_topology() on the current position.
    // If update_topology() was not called on the current position, falls back to update_move().
    void update_move_given_score(Board& board, const Move& m);

    // corresonds to set_policy_mask. Update groupIds.
    // after update_topology, a group is named by the square index of its maximal scoring move :
    // if groupId = BASEGID <-> square is either illegal, or (isn't maximal scoring move and can't be a score)
    // if groupId != BASEGID && groupId != square <-> square is legal and move groupId (maximal scoring move) makes this square a score.
    // if groupId != BASEGID && groupId == square <-> square is legal and this move is maximal scoring move.
    void update_topology(const Board& board);

    // return a list of cords which does not have superior score building move.
    // only return moves that gains equal or more points than minScore.
    std::vector<int> maximize_score(const Board& board, const Color c, const uint8_t minScore) const;

    // find a move that maximizes score which includes cord. If no move generates score, returns cord.
    int maximize_score(const Board& board, Color c, const Move& cord) const;

    std::vector<float> transfer_logit(const std::vector<float>& logit, const std::vector<int>& options, const Color turn) const;

    const float obtained_score(Color c) const;

private:
    float obtainedScore[2];

    // true iff groupIds describe the current position (update_topology() called, no move played since).
    bool topologyCurrent;
    
    // group info for black/white
    std::array<GroupId, boardSize + 1> groupIds[2];
    std::bitset<boardSize> scoreCandidates[2];

    void update_score_candidates(const Board& board, const Color c);

    void update_score(Board& board, uint8_t r, uint8_t c, const Color played);

    // calculates if territory is formed on given grid using BFS.
    // Also updates game state.
    uint8_t check_score(Board& board, uint8_t r, uint8_t c, Color clr);

    /////////////////////////////////////////////////////////
    // Functions needed for effective expansion.

    // Faster version of checkScore but segTable should be provided.
    // Calculate whether playing at {r, c} would give territory, without changing game state.
    // Returns bitset with 1 if corresponding segment becomes territory.
    std::bitset<boardSize> check_score(const Board& board, uint8_t r, uint8_t c, const SegTable& segTable) const;

    // updates scoreCandidates of both colors and returns {black segTable, white segTable} built from scratch.
    std::pair<SegTable, SegTable> update_segment_table(const Board& board);

    const std::bitset<boardSize>& score_candidates(Color c) const;

    std::bitset<boardSize>& score_candidates(Color c);

    const std::array<GroupId, boardSize + 1>& group_ids(Color c) const;

    std::array<GroupId, boardSize + 1>& group_ids(Color c);
};


#endif
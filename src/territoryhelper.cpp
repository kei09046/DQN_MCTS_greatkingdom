#include "territoryhelper.h"

#include <queue>
#include <cassert>
#include <limits>
#include <algorithm>

namespace{
    constexpr uint8_t BASEGROUPID = 255U;

    // masks used by bitboard shifts. Bit index = r * colSize + c.
    struct BoardMasks{
        std::bitset<boardSize> edge;        // squares on the board border
        std::bitset<boardSize> sides[4];    // top row, bottom row, left column, right column
    };

    const BoardMasks MASKS = []{
        BoardMasks m;
        for(int r=0; r<rowSize; ++r)
            for(int c=0; c<colSize; ++c){
                const int idx = r * colSize + c;
                m.edge[idx] = (r == 0 || c == 0 || r == rowSize - 1 || c == colSize - 1);
                m.sides[0][idx] = (r == 0);
                m.sides[1][idx] = (r == rowSize - 1);
                m.sides[2][idx] = (c == 0);
                m.sides[3][idx] = (c == colSize - 1);
            }
        return m;
    }();

    inline Color opposite(Color c){
        return (c == BLACK) ? WHITE : BLACK;
    }

    // Territory gained by every candidate move of both colors, with one DFS over the empty squares
    // (Hopcroft-Tarjan cut vertices). Filling x splits x's empty region into
    //   - the DFS subtree of each child ch of x with low[ch] >= disc[x], and
    //   - the rest of the region (region minus x minus those subtrees), if any.
    // Each piece is territory for color k if no square of it is next to an opponent stone and it doesn't touch all 4 sides,
    // the same rule as check_score(). Subtrees are kept as bitboards, so each test is a few ANDs.
    // gain[k][x] : squares color k (0 : black, 1 : white) gains by playing x. Only filled for x in cand[k].
    void compute_gains(const Board& board, const std::bitset<boardSize> (&cand)[2], std::array<std::bitset<boardSize>, boardSize> (&gain)[2]){
        using BB = std::bitset<boardSize>;
        const BB& empty = board.empty();
        const BB adjOpp[2] = {board.adjacent(WHITE), board.adjacent(BLACK)};
        const BB anyCand = cand[0] | cand[1];

        auto add_piece = [&](int x, const BB& piece){
            bool allSides = true;
            for(const auto& side : MASKS.sides)
                allSides = allSides && (piece & side).any();
            if(allSides)
                return;
            for(int k=0; k<2; ++k)
                if(cand[k][x] && (piece & adjOpp[k]).none())
                    gain[k][x] |= piece;
        };

        std::array<int, boardSize> disc, low, parent;
        std::array<uint8_t, boardSize> nextNb;      // next neighbor to visit (iterative DFS)
        std::array<BB, boardSize> sub;              // squares of the DFS subtree
        std::array<BB, boardSize> sep;              // union of subtrees separated from the region when the square is filled
        std::array<int, boardSize> stack;
        disc.fill(-1);
        int timer = 0;

        for(int root=0; root<boardSize; ++root){
            if(!empty[root] || disc[root] >= 0)
                continue;

            int top = 0;
            stack[top++] = root;
            disc[root] = low[root] = timer++;
            parent[root] = -1;
            nextNb[root] = 0;
            sub[root].reset();
            sub[root].set(root);
            sep[root].reset();

            while(top > 0){
                const int v = stack[top - 1];
                const auto& nb = NEIGHBORS4[v];
                if(nextNb[v] < nb.count){
                    const int w = to_idx(nb.cells[nextNb[v]++]);
                    if(!empty[w])
                        continue;
                    if(disc[w] < 0){
                        disc[w] = low[w] = timer++;
                        parent[w] = v;
                        nextNb[w] = 0;
                        sub[w].reset();
                        sub[w].set(w);
                        sep[w].reset();
                        stack[top++] = w;
                    }
                    else if(w != parent[v])
                        low[v] = std::min(low[v], disc[w]);
                }
                else{
                    --top;
                    const int p = parent[v];
                    if(p < 0)
                        continue;
                    low[p] = std::min(low[p], low[v]);
                    sub[p] |= sub[v];
                    // v's subtree can only reach the rest of the region through p (always true for the root's children).
                    if(low[v] >= disc[p]){
                        sep[p] |= sub[v];
                        if(anyCand[p])
                            add_piece(p, sub[v]);
                    }
                }
            }

            // the piece containing x's DFS parent side.
            const BB region = sub[root];
            const BB regionCand = region & anyCand;
            for(size_t x = regionCand._Find_first(); x < boardSize; x = regionCand._Find_next(x)){
                BB rest = region & ~sep[x];
                rest.reset(x);
                if(rest.any())
                    add_piece(static_cast<int>(x), rest);
            }
        }
    }
}

TerritoryHelper::TerritoryHelper() : obtainedScore{0.0f, 0.0f}, topologyCurrent(false){
    for(int i=0; i<2; ++i)
        groupIds[i].fill(BASEGROUPID);
}

// slow version works without update_topology. Should be used on non-branching game, resetted game etc.
void TerritoryHelper::update_move(Board& board, const Move& m){
    topologyCurrent = false;
    if(m == PASSMOVE)
        return;

    Color toCheck = board.at(m);
    assert(toCheck == BLACK || toCheck == WHITE);
    int idx = (toCheck == BLACK) ? 0 : 1;

    const BitBoard& own = board.own(toCheck);
    int cnt = 0;
    for(auto [nr, nc] : neighbors8(m.first, m.second))
        cnt += own[to_idx(nr, nc)] ? 1 : 0;

    if(cnt < (MASKS.edge[to_idx(m.first, m.second)] ? 1 : 2))
        return;

    for (auto [tr, tc] : neighbors4(m.first, m.second))
        obtainedScore[idx] += static_cast<float>(check_score(board, tr, tc, toCheck));
}

// fast version, update_topology call should be preceded which reuses previous turn's state.
void TerritoryHelper::update_move_given_score(Board& board, const Move& move){
    // groupIds are only valid for the position update_topology() was called on.
    if(!topologyCurrent){
        update_move(board, move);
        return;
    }
    topologyCurrent = false;
    if(move == PASSMOVE)
        return;

    const Color c = board.at(move);
    assert(c == BLACK || c == WHITE);
    std::vector<int> acquired;
    int moveInt = move.first * colSize + move.second;

    const auto& group = group_ids(c);
    // group == BASEGROUPID : square can't be a territory by any move.
    // group != moveInt : group was set by other move which generates territory (this move scores nothing).
    // group == moveInt : this is a maximal scoring move, all other squares of the group become a score.
    if(group[moveInt] != moveInt)
        return;
    for(int i=0; i<boardSize; ++i){
        if(i != moveInt && group[i] == moveInt)
            acquired.push_back(i);
    }

    // update territory without BFS.
    const int scoreIdx = (c == BLACK) ? 0 : 1;

    for(const auto& idx : acquired){
        board.mark_score(idx, c);
    }
    obtainedScore[scoreIdx] += acquired.size();
}


// corresonds to set_policy_mask
// Should compute groups for both colors.
// GroupId == square index => that is a move that makes all square with same GroupId a territory.
void TerritoryHelper::update_topology(const Board& board){
    update_score_candidates(board, BLACK);
    update_score_candidates(board, WHITE);

    std::array<std::bitset<boardSize>, boardSize> gain[2];
    compute_gains(board, scoreCandidates, gain);

    for(int k=0; k<2; ++k){
        const auto& cand = scoreCandidates[k];
        auto& groupIds = this->groupIds[k];
        groupIds.fill(BASEGROUPID);

        // candidates in increasing order; one lying in territory gained by an earlier scoring candidate is skipped.
        std::bitset<boardSize> acquired, scoring;
        for(size_t i = cand._Find_first(); i < boardSize; i = cand._Find_next(i)){
            if(!acquired[i] && gain[k][i].any()){
                acquired |= gain[k][i];
                scoring.set(i);
            }
        }

        // maximal scoring moves : scoring moves not inside territory of another scoring move.
        for(size_t i = scoring._Find_first(); i < boardSize; i = scoring._Find_next(i)){
            if(acquired[i])
                continue;
            groupIds[i] = static_cast<GroupId>(i);
            const auto& g = gain[k][i];
            for(size_t j = g._Find_first(); j < boardSize; j = g._Find_next(j))
                groupIds[j] = static_cast<GroupId>(i);
        }
    }
    topologyCurrent = true;
}

// find all moves that maximizes score and generates score >= minScore.
std::vector<int> TerritoryHelper::maximize_score(const Board& board, const Color c, const uint8_t minScore) const{
    std::vector<int> ret;
    const auto& group = group_ids(c);

    if(minScore == 0){
        for(int i=0; i<boardSize; ++i){
            if(board.is_empty(i) && (group[i] == BASEGROUPID || group[i] == i))
                ret.push_back(i);
        }
    }
    else{
        std::vector<int> groupSizeCntr(boardSize, 0);

        for(const auto& groupId : group)
            if(groupId != BASEGROUPID)
                groupSizeCntr[groupId]++;

        for(int i=0; i<boardSize; ++i){
            if((group[i] == i) && (groupSizeCntr[i] > minScore))
                ret.push_back(i);
        }
    }

    return ret;
}

// find a move that maximizes score which includes cord. If no move generates score, returns cord.
int TerritoryHelper::maximize_score(const Board& board, Color c, const Move& cord) const{
    const auto& group = group_ids(c);

    int cordInt = cord.first * colSize + cord.second;
    int cordGroup = group.at(cordInt);
    // if no scoring moves exist
    if(cordGroup == BASEGROUPID){
        return cordInt;
    }

    // group id is the square of its maximal scoring move.
    assert(board.is_empty(cordGroup) && group[cordGroup] == cordGroup);
    return cordGroup;
}

std::vector<float> TerritoryHelper::transfer_logit(const std::vector<float>& logit, const std::vector<int>& options, const Color turn) const{
    const auto& group = group_ids(turn);
    const int moveSize = options.size();
    // options = list of moves : [1, 6, 8, ...]
    // groupId : [0, 0, 0, 255, 255, 255, 1, 0, 2, ...] (boardSize + 1)
    // all cords with same groupId < 255 shares unique option that makes it all territory.
    // logit : size boardSize + 1

    // maps group num -> move
    std::array<int, boardSize + 1> groupToOptionIdx;
    groupToOptionIdx.fill(-1);

    for(int i=0; i<moveSize; ++i){
        int gid = group[options[i]];
        if(gid != BASEGROUPID)
            groupToOptionIdx.at(gid) = i;
    }

    std::vector<float> transferred(moveSize, std::numeric_limits<float>::lowest());
    for(int i=0; i<moveSize; ++i){
        transferred[i] = logit.at(options.at(i));
    }
    for(int i=0; i<boardSize; ++i){
        if(group[i] != BASEGROUPID && groupToOptionIdx[group[i]] >= 0){
            const int k = groupToOptionIdx[group[i]];
            transferred[k] = std::max(transferred[k], logit[i]);
        }
    }

    // add passmove logit
    if(options.back() == PASSMOVE.first * colSize + PASSMOVE.second)
        transferred.back() = logit.back();

    return transferred;
}


const float TerritoryHelper::obtained_score(Color c) const{
    return obtainedScore[(c == BLACK) ? 0 : 1];
}

void TerritoryHelper::update_score_candidates(const Board& board, const Color clr){
    const BitBoard own = board.own(clr);

    // neighbor bitboards: bit p of a bitboard is set iff that neighbor of p is in own.
    // horizontal shifts need column masks to prevent wrapping across rows;
    // vertical shifts just fall off the board. Vertical shifts keep the column, so w/e masks stay valid.
    const auto w = (own << 1) & ~MASKS.sides[2];    // not on the left column
    const auto e = (own >> 1) & ~MASKS.sides[3];    // not on the right column
    const std::bitset<boardSize> nbrs[8] = {
        w, e,
        own << colSize, own >> colSize,   // up, down
        w << colSize, w >> colSize,       // up-left, down-left
        e << colSize, e >> colSize        // up-right, down-right
    };

    // bit-parallel saturating counter: ones = count >= 1, twos = count >= 2.
    std::bitset<boardSize> ones, twos;
    for(const auto& n : nbrs){
        twos |= ones & n;
        ones |= n;
    }

    score_candidates(clr) = board.empty() & ((MASKS.edge & ones) | (~MASKS.edge & twos));
}

uint8_t TerritoryHelper::check_score(Board& board, uint8_t r, uint8_t c, Color clr) {
    if (!board.is_empty(to_idx(r, c)))
        return 0U;

    const BitBoard& adjToOppositeSide = board.adjacent(opposite(clr));
    uint8_t meetEdgeFlags = 0;  // 4bit integer to track edge touching
    std::queue<Move> q;
    BitBoard emptyCells;
    uint8_t areaCount = 0;

    q.emplace(r, c);
    std::bitset<boardSize> marker;
    marker[r * colSize + c] = true;

    while (!q.empty()) {
        auto [tr, tc] = q.front();
        q.pop();

        const int tIdx = to_idx(tr, tc);
        if (board.is_empty(tIdx)) { // if cell is empty
            if (adjToOppositeSide[tIdx]) { // if place is adjacent to opponent stone
                return 0; 
            }
            meetEdgeFlags |= (tr == 0);          // Top edge
            meetEdgeFlags |= (tr == rowSize - 1) << 1; // Bottom edge
            meetEdgeFlags |= (tc == 0) << 2;          // Left edge
            meetEdgeFlags |= (tc == colSize - 1) << 3; // Right edge

            areaCount++;
            emptyCells.set(tIdx);

            for (auto [nr, nc] : neighbors4(tr, tc)) {
                if (!marker[nr * colSize + nc]) {
                    q.emplace(nr, nc);
                    marker[nr * colSize + nc] = true;
                }
            }
        }
        // else neutral; nothing to be done
    }

    if (meetEdgeFlags == 0b1111)  // All edges are touched
        return 0U;

    // Mark valid territory
    board.mark_score(emptyCells, clr);

    return areaCount;
}

const std::bitset<boardSize> &TerritoryHelper::score_candidates(Color c) const{
    return scoreCandidates[(c == BLACK) ? 0 : 1];
}

std::bitset<boardSize> &TerritoryHelper::score_candidates(Color c){
    return scoreCandidates[(c == BLACK) ? 0 : 1];
}

const std::array<GroupId, boardSize + 1>& TerritoryHelper::group_ids(Color c) const{
    return groupIds[(c == BLACK) ? 0 : 1];
}

std::array<GroupId, boardSize + 1>& TerritoryHelper::group_ids(Color c){
    return groupIds[(c == BLACK) ? 0 : 1];
}


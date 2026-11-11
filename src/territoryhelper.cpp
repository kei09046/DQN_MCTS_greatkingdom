#include "territoryhelper.h"

#include <queue>
#include <cassert>
#include <limits>
#include <algorithm>

namespace{
    constexpr uint8_t BASEGROUPID = 255U;
    constexpr uint8_t BASESEGID = 255U;

    // masks used by bitboard shifts. Bit index = r * colSize + c.
    struct BoardMasks{
        std::bitset<boardSize> notFirstCol; // squares with c != 0
        std::bitset<boardSize> notLastCol;  // squares with c != colSize - 1
        std::bitset<boardSize> edge;        // squares on the board border
    };

    const BoardMasks MASKS = []{
        BoardMasks m;
        for(int r=0; r<rowSize; ++r)
            for(int c=0; c<colSize; ++c){
                const int idx = r * colSize + c;
                m.notFirstCol[idx] = (c != 0);
                m.notLastCol[idx] = (c != colSize - 1);
                m.edge[idx] = (r == 0 || c == 0 || r == rowSize - 1 || c == colSize - 1);
            }
        return m;
    }();

    inline Color opposite(Color c){
        return (c == BLACK) ? WHITE : BLACK;
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
    // update_segment_table also updates scoreCandidates.
    const auto segTables = update_segment_table(board);

    for(const Color c : {BLACK, WHITE}){
        const SegTable& segTable = (c == BLACK) ? segTables.first : segTables.second;
        const auto& segIds = segTable.segMap;
        auto& groupIds = group_ids(c);

        // reset groupIds
        groupIds.fill(BASEGROUPID);
        
        std::bitset<boardSize> acquiredSeg;
        std::array<std::bitset<boardSize>, boardSize> buffer;
        // push back possible moves
        std::vector<uint8_t> nonEmptySquaresIdx;
        for(uint8_t i=0U; i<boardSize; ++i){
            if(board.is_empty(i))
                nonEmptySquaresIdx.push_back(i);
        }

        for(const auto i : nonEmptySquaresIdx){
            // if(segIdx[i] < 0 || segIdx[i] >= 81){
            //     for(int j=0; j<rowSize; ++j){
            //         for(int k=0; k<colSize; ++k){
            //             std::cerr << static_cast<int>(segIdx[j * colSize + k]);
            //         }
            //         std::cerr << std::endl;
            //     }
            // }

            if(score_candidates(c).test(i) && !acquiredSeg.test(segIds[i])){
                auto acquired = check_score(board, i/colSize, i%colSize, segTable);
                acquiredSeg |= acquired;
                buffer[i] = acquired;
            }
        }

        // squares of each segment, so acquired segments can be turned into acquired squares with bitwise OR.
        std::array<std::bitset<boardSize>, boardSize> segSquares{};
        for(const auto j : nonEmptySquaresIdx)
            if(segIds[j] != BASESEGID)
                segSquares[segIds[j]].set(j);

        for(const auto i : nonEmptySquaresIdx){
            // segIds[i] = BASEID for i in nonEmptySquares <-> not adjacent to any potential score gaining moves
            if(segIds[i] == BASESEGID || !acquiredSeg.test(segIds[i])){
                // if territory is gained.
                if(!buffer[i].none()){
                    groupIds[i] = i;
                    std::bitset<boardSize> acquiredSquares;
                    for(size_t t = buffer[i]._Find_first(); t < boardSize; t = buffer[i]._Find_next(t))
                        acquiredSquares |= segSquares[t];
                    for(size_t j = acquiredSquares._Find_first(); j < boardSize; j = acquiredSquares._Find_next(j))
                        groupIds[j] = groupIds[i];
                }
            }
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
    const auto w = (own << 1) & MASKS.notFirstCol;
    const auto e = (own >> 1) & MASKS.notLastCol;
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

std::bitset<boardSize> TerritoryHelper::check_score(const Board& board, uint8_t r, uint8_t c, const SegTable& segTable) const{
    const auto& segIdx = segTable.segMap;
    const auto& segInfos = segTable.segInfos;

    uint8_t blockedSeg = segIdx[r * colSize + c];
    std::bitset<boardSize> globalVisitMark, terrMark;
    globalVisitMark[blockedSeg] = true;


    for(auto [nr, nc] : neighbors4(r, c)){
        if(!board.is_empty(to_idx(nr, nc)))
            continue;

        std::bitset<4> meetEdgeFlags;  // 4bit integer to track edge touching
        std::queue<uint8_t> segQ;
        std::bitset<boardSize> localVisitMark;
        bool isScore = true;

        const auto startSeg = segIdx[nr * colSize + nc];
        assert(blockedSeg != startSeg);

        // if segment was checked by previous iterations.
        if(globalVisitMark[startSeg])
            continue;

        segQ.push(startSeg);
        localVisitMark[blockedSeg] = true;
        localVisitMark[startSeg] = true;

        // perform BFS on segment
        while (!segQ.empty()) {
            const SegInfo& sInfo = segInfos[segQ.front()];
            segQ.pop();

            // if segment is adjacent to opposite color
            if(sInfo.adjOpp){
                isScore = false;
                break;
            }
            meetEdgeFlags |= sInfo.adjEdge;
            if(meetEdgeFlags.all()){
                isScore = false;
                break;
            }

            const std::bitset<boardSize> next = sInfo.connectedSeg & ~localVisitMark;
            if((next & globalVisitMark).any()){
                // startSeg hasn't been visited yet a reachable segment has been visited.
                // This indicates BFS quitted in previous iteration, which means this is not territory.
                isScore = false;
                break;
            }
            localVisitMark |= next;
            for(size_t t = next._Find_first(); t < boardSize; t = next._Find_next(t))
                segQ.push(static_cast<uint8_t>(t));
        }

        if(isScore){
            // std::cerr << "Territory gained " << localVisitMark << std::endl;
            terrMark |= localVisitMark;
        }
        globalVisitMark |= localVisitMark;
    }

    terrMark[blockedSeg] = false;
    return terrMark;
}

// Returns {black segTable, white segTable}.
std::pair<SegTable, SegTable> TerritoryHelper::update_segment_table(const Board& board){
    std::pair<SegTable, SegTable> ret;

    for(const Color clr : {BLACK, WHITE}){
        SegTable& segTable = (clr == BLACK) ? ret.first : ret.second;
        auto& segMap = segTable.segMap;
        auto& segInfos = segTable.segInfos;
        segMap.fill(BASESEGID);

        // update candidates immeidately as previous candidates are no longer needed.
        update_score_candidates(board, clr);
        const auto& potScore = score_candidates(clr);

        const BitBoard& adjToOpp = board.adjacent(opposite(clr));
        std::vector<Move> terrCandidate;
        uint8_t segN = 0U;

        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                if(potScore[i * colSize + j]){
                    segMap[i * colSize + j] = segN;
                    terrCandidate.push_back({i, j});
                    std::bitset<4> adjEdge;
                    if(i == 0) 
                        adjEdge[0] = true;
                    else if(i == rowSize - 1) 
                        adjEdge[1] = true;
                    if(j == 0) 
                        adjEdge[2] = true;
                    else if(j == colSize - 1) 
                        adjEdge[3] = true;
                    segInfos[segN++] = {BitBoard{}, adjToOpp[to_idx(i, j)], adjEdge};
                }
            }
        }

        for(const Move& cand : terrCandidate){
            SegInfo& currentSeg = segInfos.at(segMap[cand.first * colSize + cand.second]);
            for(auto [r, c] : neighbors4(cand)){
                if(!board.is_empty(to_idx(r, c)))
                    continue;


                if(segMap[r * colSize + c] != 255U){
                    currentSeg.connectedSeg.set(segMap[r * colSize + c]);
                }
                // if new segment is detected
                else{
                    // set up connection
                    currentSeg.connectedSeg.set(segN);

                    // start BFS
                    std::queue<Move> seg;
                    seg.push({r, c});
                    segMap[r * colSize + c] = segN;
                    std::bitset<boardSize> checker;
                    checker[r * colSize + c] = true;

                    while(!seg.empty()){
                        Move m = seg.front();
                        seg.pop();
                        if(adjToOpp[to_idx(m)])
                            segInfos[segN].adjOpp = true;
                        if(m.first == 0) 
                            segInfos[segN].adjEdge[0] = true;
                        else if(m.first == rowSize - 1) 
                            segInfos[segN].adjEdge[1] = true;
                        if(m.second == 0) 
                            segInfos[segN].adjEdge[2] = true;
                        else if(m.second == colSize - 1) 
                            segInfos[segN].adjEdge[3] = true;
                        

                        for(auto [nr, nc] : neighbors4(m)){
                            if(board.is_empty(to_idx(nr, nc)) && !checker[nr * colSize + nc]){
                                checker[nr * colSize + nc] = true;
                                uint8_t mapVal = segMap[nr * colSize + nc];
                                if(mapVal == 255U){
                                    seg.push({nr, nc});
                                    segMap[nr * colSize + nc] = segN;
                                }
                                else if(mapVal != segN){
                                    segInfos[segN].connectedSeg.set(mapVal);
                                }
                            }
                        }
                    }
                    segN++;
                }
            }
        }

    }

    return ret;
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


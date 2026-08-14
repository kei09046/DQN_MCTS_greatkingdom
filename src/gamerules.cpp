#include "gamerules.h"
#include "modelcompare.h"
#include <utility>
#include <queue>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <bitset>
#include <cassert>

namespace{
    constexpr char dr[4] = {-1, 0, 1, 0};
    constexpr char dc[4] = {0, 1, 0, -1};
}


Game::Game() : moveCount(0), possibleMoves(), pTransferGroups(boardSize + 1, -1), threat(RESIGNMOVE), winmove(RESIGNMOVE){
    uint8_t temp = 0;
    for(int i=0; i<rowSize; ++i)
        for(int j=0; j<colSize; ++j){
            board[i][j] = EMPTY;
            chains[temp] = {0, {}};
            stones[i][j] = {temp, temp};
            temp++;
        }
    
    currentTurn = BLACK;
    score[0] = 0.0f;
    score[1] = 0.0f;
    lastTwoMoves[0] = PASSMOVE;
    lastTwoMoves[1] = PASSMOVE;

    for(const auto& neutral : globalConfig.neutrals){
        board[neutral.first][neutral.second] = NEUTRAL;
    }
}

std::pair<Color, Wintype> Game::makeMove(Move move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;
    if(move == RESIGNMOVE){ // resign
        switchTurn();
        return {currentTurn, RESIGN};
    }

    if(move == PASSMOVE){ // pass
        switchTurn();
        moveCount++;
        return {EMPTY, NONE};
    }

    uint8_t r = move.first;
    uint8_t c = move.second;
    assert(board[r][c] & EMPTY);
    // turn off the empty bit, turn on the color bit.
    board[r][c] ^= currentTurn | EMPTY;
    for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            board[nr][nc] |= adjTo(currentTurn);
        }
    }

    Color clr = captureResultbyMove(r, c);
    if(clr != EMPTY)
        return {clr, CAPTURE};
    if(moveCount >= 2)
        updateScore(r, c);
    
    switchTurn();
    moveCount++;

    if(getLegalMoveCount() == 0 || moveCount > boardSize){
        return {scoreWinner(), SCORE};
    }
    return {EMPTY, NONE};
}

Color Game::makeMoveGivenScore(const Move& move){
    std::vector<int> acquired;
    int moveInt = move.first * colSize + move.second;
    for(int i=0; i<pTransferGroups.size(); ++i){
        if(i != moveInt && pTransferGroups[i] == pTransferGroups[moveInt])
            acquired.push_back(i);
    }

    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;

    if(move == PASSMOVE){ // pass
        switchTurn();
        moveCount++;
        return EMPTY;
    }

    uint8_t r = move.first;
    uint8_t c = move.second;

    assert(board[r][c] & EMPTY);
    // turn off the empty bit, turn on the color bit.
    board[r][c] ^= currentTurn | EMPTY;
    for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            board[nr][nc] |= adjTo(currentTurn);
        }
    }

    Color clr = captureResultbyMove(r, c);
    if(clr != EMPTY)
        return clr;
    
    // update territory without BFS.
    const Color scoreBit = (currentTurn == BLACK) ? BSCORE : WSCORE;
    const int scoreIdx = (currentTurn == BLACK) ? 0 : 1;

    for(const auto& idx : acquired){
        board[idx / colSize][idx % colSize] = scoreBit;
    }
    score[scoreIdx] += acquired.size();

    switchTurn();
    moveCount++;

    return EMPTY;
}

std::tuple<Color, Wintype, std::vector<float>> Game::makeMoveWithStat(Move move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;
    // if(move == RESIGNMOVE){ // If resign, find the stones that have 1 liberties + single liberty is not territory; add all of them to captureMap.
    //     std::vector<float> captureMap(boardSize, 0.0f);
    //     for(int i=0; i<rowSize; ++i){
    //         for(int j=0; j<colSize; ++j){
    //             if((board[i][j] & currentTurn)){
    //                 const Chain& c = chains[findHead(i, j)];
    //                 if(c.liberties.count() == 1){
    //                     int libIdx = c.liberties._Find_first();
    //                     if(!(board[libIdx / colSize][libIdx % colSize] & SCOREMASK))
    //                         captureMap[i * colSize + j] = 1.0f;
    //                 }
    //             }
    //         }
    //     }

    //     switchTurn();
    //     return {currentTurn, RESIGN, captureMap};
    // }

    if(move == PASSMOVE){ // pass
        switchTurn();
        moveCount++;
        return {EMPTY, NONE, {}};
    }

    uint8_t r = move.first;
    uint8_t c = move.second;
    if(!(board[r][c] & EMPTY)){
        ModelCompare::displayBoardGUI(true, *this);
        printMove(move);
    }
    assert(board[r][c] & EMPTY);
    // turn off empty bit, turn on color bit.
    board[r][c] ^= currentTurn | EMPTY;
    for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            board[nr][nc] |= adjTo(currentTurn);
        }
    }

    auto [clr, scoreMap] = captureResultWithStat(r, c);
    if(clr != EMPTY)
        return {clr, CAPTURE, scoreMap};
        
    if(moveCount >= 2)
        updateScore(r, c);
    
    switchTurn();
    moveCount++;

    if(getLegalMoveCount() == 0 || moveCount > boardSize){
        std::vector<float> scoreMap(boardSize);
        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                scoreMap[i * colSize + j] = (board[i][j] & (BSCORE | BLACK)) ? -1.0f : (board[i][j] & (WSCORE | WHITE) ? 1.0f : 0.0f);
            }
        }
        return {scoreWinner(), SCORE, scoreMap};
    }
    return {EMPTY, NONE, {}};
}


void Game::setPolicyMask(){
    possibleMoves.clear();
    std::fill(pTransferGroups.begin(), pTransferGroups.end(), -1);

    const auto& [tactic, wdl] = tacticCheck();
    if(wdl > 0){
        winmove = tactic;
        possibleMoves.push_back(winmove.first * colSize + winmove.second);
        pTransferGroups.at(winmove.first * colSize + winmove.second) = 0;
        return;
    }
    else if(wdl < 0){
        threat = tactic;
        possibleMoves.push_back(PASSMOVE.first * colSize + PASSMOVE.second);
        pTransferGroups.at(PASSMOVE.first * colSize + PASSMOVE.second) = 0;
        return;
    }
    else{
        threat = tactic;
        std::bitset<boardSize> potScore;
        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                potScore.set(i * colSize + j, (board[i][j] & EMPTY) && canbeScore(i, j, currentTurn));
            }
        }

        if(threat != RESIGNMOVE){
            // first find list of possible moves.
            std::vector<int> candidates = candidatesWhenThreat(potScore);
            // std::cerr << "possible moves : " << std::endl;
            // for(const auto& m : possibleMoves)
            //     printMove(m);

            // now, consider all options in possibleMoves. If none of them makes T territory, should play at T.
            const auto segTable = segmentTable(potScore);
            const auto& segIdx = segTable.second;
            // for(int i=0; i<rowSize; ++i){
            //     for(int j=0; j<colSize; ++j){
            //         std::cerr << static_cast<int>(segTable.second[i * colSize + j]) << " ";
            //     }
            //     std::cerr << std::endl;
            // }

            const auto threatSeg = segIdx.at(threat.first * colSize + threat.second);
            std::bitset<boardSize> acquiredSeg, threatAcquireSeg;

            int threatInt = threat.first * colSize + threat.second;
            // if only possible move is to play at threat, and if it can't generate any score
            if(candidates.size() == 1 && segIdx[threatInt] == 255U){
                assert(candidates[0] == threatInt);
                possibleMoves.push_back(threatInt);
                pTransferGroups.at(threatInt) = 0;
                return;
            }

            else{
                for(const auto& option : candidates){
                    const std::bitset<boardSize> acquired = checkScore(option / colSize, option % colSize, currentTurn, segTable);
                    acquiredSeg |= acquired;
                    // if option == threat, it means that no other move made threat a territory.
                    if(acquired[threatSeg] || (option == threatInt)){
                        possibleMoves.push_back(option);
                        pTransferGroups.at(option) = 0;
                        threatAcquireSeg = acquired;
                        // can break here. Not a bug.
                        break;
                    }
                }
            }

            if(!threatAcquireSeg.none()){
                for(uint8_t i=0U; i<boardSize; ++i){
                    if(segIdx[i] != 255U && threatAcquireSeg.test(segIdx[i]))
                        pTransferGroups[i] = 0;
                }
            }
            return;
        }

        else{
            int groupCntr = -1;
            const auto segTable = segmentTable(potScore);
            const auto& segIdx = segTable.second;
            std::bitset<boardSize> acquiredSeg;

            std::array<std::bitset<boardSize>, boardSize> buffer;
            // push back possible moves
            std::vector<uint8_t> nonEmptySquaresIdx;
            for(uint8_t i=0U; i<boardSize; ++i){
                if(isLegal(i / colSize, i % colSize))
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

                if(potScore.test(i) && !acquiredSeg.test(segIdx[i])){
                    auto acquired = checkScore(i/colSize, i%colSize, currentTurn, segTable);
                    acquiredSeg |= acquired;
                    buffer[i] = acquired;
                }
            }

            for(const auto i : nonEmptySquaresIdx){
                // segIdx[i] = 255 for i in nonEmptySquares <-> not adjacent to any potential score gaining moves
                if(segIdx[i] == 255U || !acquiredSeg.test(segIdx[i])){
                    possibleMoves.push_back(i);
                    pTransferGroups.at(i) = ++groupCntr;

                    // if territory is gained.
                    if(!buffer[i].none()){
                        for(const auto j : nonEmptySquaresIdx){
                            if((segIdx[j] != 255U) && buffer[i].test(segIdx[j]))
                                pTransferGroups.at(j) = groupCntr;
                        }
                    }
                }
            }
            if(scoreWinner() == currentTurn){
                possibleMoves.push_back(PASSMOVE.first * colSize + PASSMOVE.second);
                pTransferGroups.at(boardSize) = ++groupCntr;
            }
        }
    }
}

std::pair<Move, int> Game::tacticCheck() const{
    Move threat = RESIGNMOVE;

    std::bitset<boardSize> chainChecker;
    for(int i=0; i<rowSize; ++i){            
        for(int j=0; j<colSize; ++j){
            const Chain c = getChain({i, j});
            if(c.size != 0 && c.liberties.count() == 1 && !chainChecker.test(getChainIdx({i, j}))){
                chainChecker.set(getChainIdx({i, j}), true);
                int onlyLib = c.liberties._Find_first();

                if(isLegal(onlyLib / colSize, onlyLib % colSize)){
                    // if my stone is under threat -> have to find only move unless can capture opponent's stone.
                    if(board[i][j] & currentTurn){
                        threat = {onlyLib / colSize, onlyLib % colSize};
                    }

                    // if opponent stone is capturable
                    else{
                        return {{onlyLib / colSize, onlyLib % colSize}, 1};
                    }
                }
            }
        }
    }

    if(threat == RESIGNMOVE)
        return {RESIGNMOVE, 0};

    // check if playing at threat would extend it's liberties
    std::bitset<boardSize> liberties;
    for(int i=0; i<4; ++i){
        uint8_t r = threat.first + dr[i];
        uint8_t c = threat.second + dc[i];
        if(inbound(r, c)){
            if(board[r][c] & EMPTY){
                return {threat, 0};    
            }
            else if(board[r][c] & currentTurn){
                liberties |= chains[findHead(r, c)].liberties;
            }
        }
    }
    liberties[threat.first * colSize + threat.second] = false;

    assert(threat != RESIGNMOVE);
    if(liberties.none())
        return {threat, -1};
    return {threat, 0};
}

// std::tuple<std::pair<Move, int>, std::vector<Move>, std::vector<std::vector<uint8_t>>> Game::expand(const Move threat) const{
//     std::bitset<boardSize> potScore;
//     for(int i=0; i<rowSize; ++i){
//         for(int j=0; j<colSize; ++j){
//             potScore.set(i * colSize + j, (board[i][j] & EMPTY) && canbeScore(i, j, currentTurn));
//         }
//     }

//     if(threat != RESIGNMOVE){
//         // first find list of possible moves.

//         std::vector<Move> possibleMoves = possibleMovesWhenThreat(threat, potScore);
//         // std::cerr << "possible moves : " << std::endl;
//         // for(const auto& m : possibleMoves)
//         //     printMove(m);

//         // now, consider all options in possibleMoves. If none of them makes T territory, should play at T.
//         const auto segTable = segmentTable(potScore);
//         const auto& segIdx = segTable.second;
//         // for(int i=0; i<rowSize; ++i){
//         //     for(int j=0; j<colSize; ++j){
//         //         std::cerr << static_cast<int>(segTable.second[i * colSize + j]) << " ";
//         //     }
//         //     std::cerr << std::endl;
//         // }

//         const auto threatSeg = segIdx.at(threat.first * colSize + threat.second);
//         std::bitset<boardSize> acquiredSeg, threatAcquireSeg;

//         Move bestMove;
//         std::vector<std::vector<uint8_t>> transferTable(1, std::vector<uint8_t>());

//         // if only possible move is to play at threat, and if it can't generate any score
//         if(possibleMoves.size() == 1 && segIdx[threat.first * colSize + threat.second] == 255U){
//             assert(possibleMoves[0] == threat);
//             bestMove = threat;
//         }

//         else{
//             for(const auto& option : possibleMoves){
//                 const std::bitset<boardSize> acquired = checkScore(option.first, option.second, currentTurn, segTable);
//                 acquiredSeg |= acquired;
//                 // if option == threat, it means that no other move made threat a territory.
//                 if(acquired[threatSeg] || (option == threat)){
//                     bestMove = option;
//                     threatAcquireSeg = acquired;
//                     // can break here. Not a bug.
//                     break;
//                 }
//             }
//         }

//         if(!threatAcquireSeg.none()){
//             for(uint8_t i=0U; i<boardSize; ++i){
//                 if(segIdx[i] != 255U && threatAcquireSeg.test(segIdx[i]))
//                     transferTable[0].push_back(i);
//             }
//         }

//         assert(transferTable.size() == 1);
//         return {{bestMove, 0}, {bestMove}, transferTable};
//     }

//     else{
//         const auto segTable = segmentTable(potScore);
//         const auto& segIdx = segTable.second;
//         std::bitset<boardSize> acquiredSeg;
//         std::vector<std::vector<uint8_t>> transferTable;
//         std::array<std::bitset<boardSize>, boardSize> buffer;

//         // push back possible moves
//         std::vector<uint8_t> nonEmptySquaresIdx;
//         for(uint8_t i=0U; i<boardSize; ++i){
//             if(isLegal(i / colSize, i % colSize))
//                 nonEmptySquaresIdx.push_back(i);
//         }

//         for(const auto i : nonEmptySquaresIdx){
//             // if(segIdx[i] < 0 || segIdx[i] >= 81){
//             //     for(int j=0; j<rowSize; ++j){
//             //         for(int k=0; k<colSize; ++k){
//             //             std::cerr << static_cast<int>(segIdx[j * colSize + k]);
//             //         }
//             //         std::cerr << std::endl;
//             //     }
//             // }

//             if(potScore.test(i) && !acquiredSeg.test(segIdx[i])){
//                 auto acquired = checkScore(i/colSize, i%colSize, currentTurn, segTable);
//                 acquiredSeg |= acquired;
//                 buffer[i] = acquired;
//             }
//         }

//         std::vector<Move> possibleMoves;
//         for(const auto i : nonEmptySquaresIdx){
//             // segIdx[i] = 255 for i in nonEmptySquares <-> not adjacent to any potential score gaining moves
//             if(segIdx[i] == 255U || !acquiredSeg.test(segIdx[i])){
//                 possibleMoves.push_back({i / colSize, i % colSize});
//                 transferTable.push_back({});

//                 // if territory is gained.
//                 if(!buffer[i].none()){
//                     for(const auto j : nonEmptySquaresIdx){
//                         if((segIdx[j] != 255U) && buffer[i].test(segIdx[j]))
//                             transferTable.back().push_back(j);
//                     }
//                 }
//             }
//         }
//         if(scoreWinner() == currentTurn){
//             possibleMoves.push_back(PASSMOVE);
//             transferTable.push_back({});
//         }
        
//         return {{RESIGNMOVE, 0}, possibleMoves, transferTable};
//     }

//     //std::cerr << "expand finished" << std::endl;
//     #ifdef measureTime
//     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//     expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//     #endif
// }

void Game::onGameEnd(Color winner){
    std::cout << "game over! winner is : " << static_cast<int>(winner) << std::endl;
}


void Game::mergeChains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2) {
    //std::cout << "merging chain :" << (int)r1 << (int)c1 << " " << (int)r2 << (int)c2 << std::endl;
    uint8_t h1 = findHead(r1, c1), h2 = findHead(r2, c2);
    if (h1 == h2) return;
    
    chains[h1].liberties.set(r2 * colSize + c2, false);
    chains[h2].liberties.set(r1 * colSize + c1, false);

    if (chains[h1].size < chains[h2].size) std::swap(h1, h2);
    chains[h1].size += chains[h2].size;
    chains[h1].liberties |= chains[h2].liberties;
    
    uint8_t cur = h2, start = h2;
    do {
        stones[cur / colSize][cur % colSize].head = h1;
        cur = stones[cur / colSize][cur % colSize].next;
    } while (cur != start);
    
    std::swap(stones[h2 / colSize][h2 % colSize].next, stones[h1 / colSize][h1 % colSize].next);
    //std::cout << "after liberties :\n " << chains[h1].liberties << std::endl;
}


Color Game::captureResultbyMove(uint8_t r, uint8_t c){
    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {1U, {}};
    Color sColor = getBoard({r, c});

    for (int i = 0; i < 4; ++i) {
        uint8_t nr = r + dr[i], nc = c + dc[i];
        if (!inbound(nr, nc)) continue;

        Color adjColor = getBoard({nr, nc});
        if(adjColor == EMPTY){ 
            chains[findHead(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (adjColor == sColor){
            mergeChains(r, c, nr, nc);
        }
        else if (adjColor == reverseColor(sColor)){ 
            auto& adj_chain = chains[findHead(nr, nc)];
            adj_chain.liberties.set(r * colSize + c, false);
            if(adj_chain.liberties.none())
                return sColor;
        }
        // else adjacent to neutral; nothing should happen
    }

    if(chains[findHead(r, c)].liberties.none())
        return reverseColor(sColor);
    
    return EMPTY;
}

std::pair<Color, std::vector<float>> Game::captureResultWithStat(uint8_t r, uint8_t c){
    Color winner = EMPTY;
    std::vector<uint8_t> capturedChainIdx;

    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {1U, {}};
    Color sColor = getBoard({r, c});

    for (int i = 0; i < 4; ++i) {
        uint8_t nr = r + dr[i], nc = c + dc[i];
        if (!inbound(nr, nc)) continue;

        Color adjColor = getBoard({nr, nc});
        if(adjColor == EMPTY){ 
            chains[findHead(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (adjColor == sColor){
            mergeChains(r, c, nr, nc);
        }
        else if (adjColor == reverseColor(sColor)){ 
            auto& adj_chain = chains[findHead(nr, nc)];
            adj_chain.liberties.set(r * colSize + c, false);
            if(adj_chain.liberties.none()){
                winner = sColor;
                capturedChainIdx.push_back(findHead(nr, nc));
            }
        }
        // else adjacent to neutral; nothing should happen
    }

    if(winner == EMPTY && chains[findHead(r, c)].liberties.none()){
        winner = reverseColor(sColor);
        capturedChainIdx.push_back(findHead(r, c));
    }

    if(winner != EMPTY){
        std::vector<float> scoreMap(boardSize, 0.0f);

        for(auto idx : capturedChainIdx){
            int v = (winner == BLACK) ? 0 : 1;
            uint8_t repl = (winner == BLACK) ? BSCORE : WSCORE;

            score[v] += chains[idx].size;

            uint8_t cur = idx, start = idx;
            do {
                board[cur / colSize][cur % colSize] = repl;
                cur = stones[cur / colSize][cur % colSize].next;
            } while (cur != start);
        }

        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                scoreMap[i * colSize + j] = (board[i][j] & (BSCORE | BLACK)) ? -1.0f : (board[i][j] & (WSCORE | WHITE) ? 1.0f : 0.0f);
            }
        }

        return {winner, scoreMap};
    }

    return {EMPTY, {}};
}


//////////////////////////////////////////////////////////////////////////////////
/// Score related functions

void Game::updateScore(uint8_t r, uint8_t c) { // major bottleneck
    Color toCheck = getBoard({r, c});
    int idx = (toCheck == BLACK) ? 0 : 1;
    if(!canbeScore(r, c, toCheck))
        return;

    for (int i = 0; i < 4; ++i) {
        uint8_t tr = r + dr[i], tc = c + dc[i];
        score[idx] += static_cast<float>(checkScore(tr, tc, toCheck));
    }
}

uint8_t Game::checkScore(uint8_t r, uint8_t c, Color clr) {
    if (!(inbound(r, c) && (board[r][c] & EMPTY)))
        return 0U;

    uint8_t adjToOppositeSide = adjToOpposite(clr);
    uint8_t meetEdgeFlags = 0;  // 4bit integer to track edge touching
    std::queue<Move> q;
    std::vector<Move> emptyCells;
    uint8_t areaCount = 0;

    q.emplace(r, c);
    std::bitset<boardSize> marker;
    marker[r * colSize + c] = true;

    while (!q.empty()) {
        auto [tr, tc] = q.front();
        q.pop();

        if (board[tr][tc] & EMPTY) { // if cell is empty
            if (board[tr][tc] & adjToOppositeSide) { // if place is adjacent to opponent stone
                return 0; 
            }
            meetEdgeFlags |= (tr == 0);          // Top edge
            meetEdgeFlags |= (tr == rowSize - 1) << 1; // Bottom edge
            meetEdgeFlags |= (tc == 0) << 2;          // Left edge
            meetEdgeFlags |= (tc == colSize - 1) << 3; // Right edge

            areaCount++;
            emptyCells.emplace_back(tr, tc);

            for (int i = 0; i < 4; ++i) {
                uint8_t nr = tr + dr[i], nc = tc + dc[i];
                if (inbound(nr, nc) && !marker[nr * colSize + nc]) {
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
    Color scoreMarker = (clr == BLACK) ? BSCORE : WSCORE;
    for (auto [tr, tc] : emptyCells) {
        board[tr][tc] = scoreMarker;
    }

    return areaCount;
}

std::bitset<boardSize> Game::checkScore(uint8_t r, uint8_t c, Color clr, const std::pair<std::vector<SegInfo>, std::array<uint8_t, boardSize>>& segTable) const{
    if (!(inbound(r, c) && (board[r][c] & EMPTY) && canbeScore(r, c, currentTurn)))
        return {};

    const auto& segInfos = segTable.first;
    const auto& segIdx = segTable.second;

    uint8_t blockedSeg = segIdx[r * colSize + c];
    std::bitset<boardSize> globalVisitMark, terrMark;
    globalVisitMark[blockedSeg] = true;


    for(int i=0; i<4; ++i){
        auto nr = r + dr[i];
        auto nc = c + dc[i];
        if(!(inbound(nr, nc) && (board[nr][nc] & EMPTY)))
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
            const SegInfo sInfo = segInfos[segQ.front()];
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

            for(const uint8_t adjSegs : sInfo.connectedSeg){
                if(localVisitMark[adjSegs])
                    continue;

                if(globalVisitMark[adjSegs]){
                    // if flow reaches here, it basically means that startSeg hasn't been visited yet reachable segment has been visited.
                    // This indicates BFS quitted in previous iteration, which means this is not territory.
                    isScore = false;
                    break;
                }
                else{
                    segQ.push(adjSegs);
                    localVisitMark[adjSegs] = true;
                }
            }
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

bool Game::canbeScore(uint8_t r, uint8_t c, Color clr) const{
    int cnt = 0;
    for (int dr = ((r == 0U) ? 0 : -1); dr <= ((r == rowSize-1) ? 0 : 1); dr++)
        for (int dc = ((c == 0U) ? 0 : -1); dc <= ((c == colSize-1) ? 0 : 1); dc++)
                cnt += ((board[r+dr][c+dc] & (clr | NEUTRAL)) ? 1 : 0);
    
    if(board[r][c] & (clr | NEUTRAL))
        cnt--;

    return cnt >= ((r == 0U || c == 0U || r == rowSize-1 || c == colSize-1) ? 1 : 2);
}

std::pair<std::vector<SegInfo>, std::array<uint8_t, boardSize>> Game::segmentTable(const std::bitset<boardSize>& potScore) const{
    std::array<uint8_t, boardSize> segMap;
    std::vector<SegInfo> segInfos;
    // maximum possible number of segments
    // Absolutely essential to prevent memory reallocation.
    segInfos.reserve(boardSize - moveCount);
    segMap.fill(255U);

    std::vector<Move> terrCandidate;
    uint8_t segN = 0U;

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            if(potScore[i * colSize + j]){
                segMap[i * colSize + j] = segN++;
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
                segInfos.emplace_back(std::vector<uint8_t>{}, (board[i][j] & adjToOpposite(currentTurn)) ? 1U : 0U, std::move(adjEdge));
            }
        }
    }

    for(const Move& cand : terrCandidate){
        SegInfo& currentSeg = segInfos.at(segMap[cand.first * colSize + cand.second]);
        for(int i=0; i<4; ++i){
            uint8_t r = cand.first + dr[i];
            uint8_t c = cand.second + dc[i];
            if(!(inbound(r, c) && (board[r][c] & EMPTY)))
                continue;


            if(segMap[r * colSize + c] != 255U){
                currentSeg.connectedSeg.push_back(segMap[r * colSize + c]);
            }
            // if new segment is detected
            else{
                // set up connection
                currentSeg.connectedSeg.push_back(segN);
                segInfos.emplace_back();

                // start BFS
                std::queue<Move> seg;
                seg.push({r, c});
                segMap[r * colSize + c] = segN;
                std::bitset<boardSize> checker;
                checker[r * colSize + c] = true;

                while(!seg.empty()){
                    Move m = seg.front();
                    seg.pop();
                    if(board[m.first][m.second] & adjToOpposite(currentTurn))
                        segInfos[segN].adjOpp = true;
                    if(m.first == 0) 
                        segInfos[segN].adjEdge[0] = true;
                    else if(m.first == rowSize - 1) 
                        segInfos[segN].adjEdge[1] = true;
                    if(m.second == 0) 
                        segInfos[segN].adjEdge[2] = true;
                    else if(m.second == colSize - 1) 
                        segInfos[segN].adjEdge[3] = true;
                    

                    for(int j=0; j<4; ++j){
                        uint8_t nr = m.first + dr[j];
                        uint8_t nc = m.second + dc[j];
                        if(inbound(nr, nc) && (board[nr][nc] & EMPTY) && !checker[nr * colSize + nc]){
                            checker[nr * colSize + nc] = true;
                            uint8_t mapVal = segMap[nr * colSize + nc];
                            if(mapVal == 255U){
                                seg.push({nr, nc});
                                segMap[nr * colSize + nc] = segN;
                            }
                            else if(mapVal != segN){
                                segInfos[segN].connectedSeg.push_back(mapVal);
                            }
                        }
                    }
                }
                segN++;
            }
        }
    }

    return {segInfos, segMap};
}

std::vector<int> Game::candidatesWhenThreat(const std::bitset<boardSize>& potScore) const{
    std::vector<int> candidates;
    if(potScore.none()){
        candidates = {threat.first * colSize + threat.second};
        return candidates;
    }
        
    // find shortest path from T to opposite color stone.
    uint8_t direction[rowSize][colSize] = {0U};
    std::queue<Move> q;
    Move oppStone = RESIGNMOVE;
    Move option = RESIGNMOVE;
    
    // direction : stores way to the closest white stone. 
    direction[threat.first][threat.second] = 5U;
    q.push(threat);
    while(!q.empty()){
        Move m = q.front();
        q.pop();
        for(int i=0; i<4; ++i){
            auto nr = m.first + dr[i];
            auto nc = m.second + dc[i];
            if(inbound(nr, nc) && (board[nr][nc] & EMPTY)){
                if(board[nr][nc] & adjToOpposite(currentTurn)){
                    q = {};
                    oppStone = {nr, nc};
                    direction[nr][nc] = i+1;
                    break;
                }
                else if(direction[nr][nc] == 0U){
                    q.push({nr, nc});
                    direction[nr][nc] = i+1;
                }
            }
        }
    }

    // extremely rare case where there are no reachable opposite color stone yet it is not territory.
    if(oppStone == RESIGNMOVE){
        for(uint8_t i=0U; i<rowSize; ++i){
            for(uint8_t j=0U; j<colSize; ++j){
                if(direction[i][j] > 0U && potScore[i * colSize + j])
                    candidates.push_back(i * colSize + j);
            }
        }

        // if threat is not inserted, insert threat.
        if(!(direction[threat.first][threat.second] > 0U && potScore[threat.first * colSize + threat.second])){
            candidates.push_back(threat.first * colSize + threat.second);
        }
    }

    // for each cord in path, should check if making a move there would make T a territory.
    else{
        option = oppStone;
        while(option != threat){
            if(potScore[option.first * colSize + option.second])
                candidates.push_back(option.first * colSize + option.second);

            const auto dir = direction[option.first][option.second];
            assert(dir >= 1 && dir <= 4);
            option.first -= dr[dir - 1];
            option.second -= dc[dir - 1];
        }
        candidates.push_back(threat.first * colSize + threat.second);
    }
    return candidates;
}

uint8_t Game::getLegalMoveCount() const{
    uint8_t ret = 0;
    for(int i=0; i<rowSize; ++i)
        for(int j=0; j<colSize; ++j)
            ret += isLegal(i, j) ? 1 : 0;
    
    return ret;
}
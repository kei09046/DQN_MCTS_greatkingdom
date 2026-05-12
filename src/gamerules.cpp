#include "gamerules.h"

constexpr char dr[4] = {-1, 0, 1, 0};
constexpr char dc[4] = {0, 1, 0, -1};


Game::Game() : visitId(0), moveCount(0), finalScore(0.0f) {
    uint8_t temp = 0;
    for(int i=0; i<rowSize; ++i)
        for(int j=0; j<colSize; ++j){
            board[i][j] = EMPTY;
            scoreBoard[i][j] = EMPTY;
            mark[i][j] = 0;
            chains[temp] = {0, {}};
            stones[i][j] = {temp, temp};
            temp++;
        }
    
    currentTurn = BLACK;
    score[BLACK] = 0.0f;
    score[WHITE] = 0.0f;
    lastTwoMoves[0] = PASSMOVE;
    lastTwoMoves[1] = PASSMOVE;

    for(const auto& neutral : globalConfig.neutrals){
        board[neutral.first][neutral.second] = NEUTRAL;
        scoreBoard[neutral.first][neutral.second] = NEUTRAL;
    }
}

void Game::mergeChains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2) {
    //std::cout << "merging chain :" << (int)r1 << (int)c1 << " " << (int)r2 << (int)c2 << std::endl;
    uint8_t h1 = findHead(r1, c1), h2 = findHead(r2, c2);
    //std::cout << "head :" << (int)h1 << " " << (int)h2 << std::endl;
    //std::cout << "liberties :\n " << chains[h1].liberties << "\n" << chains[h2].liberties << std::endl;
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

    for (int i = 0; i < 4; ++i) {
        uint8_t nr = r + dr[i], nc = c + dc[i];
        if (!inbound(nr, nc)) continue;

        if(board[nr][nc] == EMPTY){ 
            chains[findHead(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (board[nr][nc] == board[r][c]){
            mergeChains(r, c, nr, nc);
        }
        else if (board[nr][nc] == reverseColor(board[r][c])){ 
            auto& adj_chain = chains[findHead(nr, nc)];
            adj_chain.liberties.set(r * colSize + c, false);
            if(adj_chain.liberties.none())
                return board[r][c];
        }
        // else adjacent to neutral; nothing should happen
    }

    if(chains[findHead(r, c)].liberties.none())
        return reverseColor(board[r][c]);
    
    return EMPTY;
}

std::pair<Color, std::vector<float>> Game::captureResultWithStat(uint8_t r, uint8_t c){
    Color winner = EMPTY;
    std::vector<uint8_t> capturedChainIdx;

    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {1U, {}};

    for (int i = 0; i < 4; ++i) {
        uint8_t nr = r + dr[i], nc = c + dc[i];
        if (!inbound(nr, nc)) continue;

        if(board[nr][nc] == EMPTY){ 
            chains[findHead(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (board[nr][nc] == board[r][c]){
            mergeChains(r, c, nr, nc);
        }
        else if (board[nr][nc] == reverseColor(board[r][c])){ 
            auto& adj_chain = chains[findHead(nr, nc)];
            adj_chain.liberties.set(r * colSize + c, false);
            if(adj_chain.liberties.none()){
                winner = board[r][c];
                capturedChainIdx.push_back(findHead(nr, nc));
            }
        }
        // else adjacent to neutral; nothing should happen
    }

    if(winner == EMPTY && chains[findHead(r, c)].liberties.none()){
        winner = reverseColor(board[r][c]);
        capturedChainIdx.push_back(findHead(r, c));
    }

    if(winner != EMPTY){
        std::vector<float> captureMap(boardSize, 0.0f);
        for(auto idx : capturedChainIdx){
            uint8_t cur = idx, start = idx;
            do {
                captureMap[cur] = 1.0f;
                cur = stones[cur / colSize][cur % colSize].next;
            } while (cur != start);
        }
        return {winner, captureMap};
    }
    
    return {EMPTY, {}};
}

uint8_t Game::checkScore(uint8_t r, uint8_t c, Color clr) {
    if (!(inbound(r, c) && (scoreBoard[r][c] & EMPTY)))
        return 0;

    uint8_t adjToOppositeSide = adjToOpposite(clr);
    uint8_t meetEdgeFlags = 0;  // 4bit integer to track edge touching
    std::queue<std::pair<uint8_t, uint8_t>> q;
    std::vector<std::pair<uint8_t, uint8_t>> emptyCells;
    uint8_t areaCount = 0;

    q.emplace(r, c);
    mark[r][c] = ++visitId;

    while (!q.empty()) {
        auto [tr, tc] = q.front();
        q.pop();

        if (scoreBoard[tr][tc] & EMPTY) { // if cell is empty
            if (scoreBoard[tr][tc] & adjToOppositeSide) { // if place is adjacent to opponent stone
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
                if (inbound(nr, nc) && mark[nr][nc] != visitId) {
                    q.emplace(nr, nc);
                    mark[nr][nc] = visitId;
                }
            }
        }
        // else neutral; nothing to be done
    }

    if (meetEdgeFlags == 0b1111)  // All edges are touched
        return 0;

    // Mark valid territory
    for (auto [tr, tc] : emptyCells) {
        scoreBoard[tr][tc] = clr;
    }

    return areaCount;
}

bool Game::canbeScore(uint8_t r, uint8_t c, Color clr){
    int cnt = 0;
    for (int dr = ((r == 0U) ? 0 : -1); dr <= ((r == rowSize-1) ? 0 : 1); dr++)
        for (int dc = ((c == 0U) ? 0 : -1); dc <= ((c == colSize-1) ? 0 : 1); dc++)
                cnt += ((board[r+dr][c+dc] == clr || board[r+dr][c+dc] == NEUTRAL) ? 1 : 0);

    return cnt >= ((r == 0U || c == 0U || r == rowSize-1 || c == colSize-1) ? 2 : 3);
}

void Game::updateScore(uint8_t r, uint8_t c) { // major bottleneck
    Color toCheck = board[r][c];
    if(!canbeScore(r, c, toCheck))
        return;

    for (int i = 0; i < 4; ++i) {
        uint8_t tr = r + dr[i], tc = c + dc[i];
        score[toCheck] += static_cast<float>(checkScore(tr, tc, toCheck));
    }
    finalScore = score[BLACK] - score[WHITE] - globalConfig.komi;
}

void Game::getScore(){
    std::queue<std::pair<uint8_t, uint8_t>> q;
    std::vector<std::pair<uint8_t, uint8_t>> emptyCells;

    for(int clr = 0; clr < 2; ++clr)
        for(int r = 0; r<rowSize; ++r)
            for(int c = 0; c<colSize; ++c){
                if (!(scoreBoard[r][c] & EMPTY))
                    continue;
            
                uint8_t adjToOppositeSide = adjToOpposite(clr);
                char meetEdgeFlags = 0;
                uint8_t areaCount = 0;

                emptyCells.clear();
                q.emplace(r, c);
                mark[r][c] = ++visitId;
            
                while (!q.empty()) {
                    auto [tr, tc] = q.front();
                    q.pop();
            
                    if (scoreBoard[tr][tc] & adjToOppositeSide)
                        continue;
            
                    if (scoreBoard[tr][tc] & EMPTY) {
                        meetEdgeFlags |= (tr == 0);          // Top edge
                        meetEdgeFlags |= (tr == rowSize - 1) << 1; // Bottom edge
                        meetEdgeFlags |= (tc == 0) << 2;          // Left edge
                        meetEdgeFlags |= (tc == colSize - 1) << 3; // Right edge
            
                        areaCount++;
                        emptyCells.emplace_back(tr, tc);
            
                        for (uint8_t i = 0; i < 4; ++i) {
                            uint8_t nr = tr + dr[i], nc = tc + dc[i];
                            if (inbound(nr, nc) && mark[nr][nc] != visitId) {
                                q.emplace(nr, nc);
                                mark[nr][nc] = visitId;
                            }
                        }
                    }
                }
            
                if (meetEdgeFlags == 0b1111) 
                    continue;
            
                // Mark valid territory
                for (auto [tr, tc] : emptyCells) {
                    scoreBoard[tr][tc] = static_cast<Color>(clr);
                }
                score[clr] += areaCount;
            }
            
    finalScore = score[BLACK] - score[WHITE] - globalConfig.komi;
}

Color Game::gameEnd(){
    return finalScore > 0 ? BLACK : WHITE;
}

uint8_t Game::getLegalMoveCount() const{
    uint8_t ret = 0;
    for(int i=0; i<rowSize; ++i)
        for(int j=0; j<colSize; ++j)
            ret += isLegal(i, j) ? 1 : 0;
    
    return ret;
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
    // update board & scoreBoard
    board[r][c] = currentTurn;
    scoreBoard[r][c] = NEUTRAL; // works as if neutral stone
    for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            scoreBoard[nr][nc] |= adjTo(currentTurn);
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
        return {gameEnd(), SCORE};
    }
    return {EMPTY, NONE};
}

std::tuple<Color, Wintype, std::vector<float>> Game::makeMoveWithStat(Move move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;
    if(move == RESIGNMOVE){ // If resign, find the stones that have 1 liberties; add all of them to captureMap.
        std::vector<float> captureMap(boardSize, 0.0f);
        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                captureMap[i * colSize + j] = (board[i][j] == currentTurn && chains[findHead(i, j)].liberties.count() == 1) ? 1.0f : 0.0f;
            }
        }

        switchTurn();
        return {currentTurn, RESIGN, captureMap};
    }

    if(move == PASSMOVE){ // pass
        switchTurn();
        moveCount++;
        return {EMPTY, NONE, {}};
    }

    uint8_t r = move.first;
    uint8_t c = move.second;
    // update board & scoreBoard
    board[r][c] = currentTurn;
    scoreBoard[r][c] = NEUTRAL; // works as if neutral stone
    for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            scoreBoard[nr][nc] |= adjTo(currentTurn);
        }
    }

    auto [clr, captureMap] = captureResultWithStat(r, c);
    if(clr != EMPTY)
        return {clr, CAPTURE, captureMap};
    if(moveCount >= 2)
        updateScore(r, c);
    
    switchTurn();
    moveCount++;

    if(getLegalMoveCount() == 0 || moveCount > boardSize){
        std::vector<float> scoreMap(boardSize);
        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                scoreMap[i * colSize + j] = (scoreBoard[i][j] == BLACK) ? -1.0f : (scoreBoard[i][j] == WHITE ? 1.0f : 0.0f);
            }
        }
        return {gameEnd(), SCORE, scoreMap};
    }
    return {EMPTY, NONE, {}};
}

// Color Game::makeMoveNoScoreUpdate(Move move){
//     lastTwoMoves[0] = lastTwoMoves[1];
//     lastTwoMoves[1] = move;

//     if(move == RESIGNMOVE){ // resign
//         switchTurn();
//         return currentTurn;
//     }

//     if(move == PASSMOVE){ // pass
//         switchTurn();
//         moveCount++;
//         return EMPTY;
//     }

//     uint8_t r = move.first;
//     uint8_t c = move.second;
//     // update board & scoreBoard
//     board[r][c] = currentTurn;
//     scoreBoard[r][c] = NEUTRAL; // works as if neutral stone

//     for(int i=0; i<4; ++i){ // make sure it can't be used for opponent
//         uint8_t nr = r + dr[i];
//         uint8_t nc = c + dc[i];
//         if(inbound(nr, nc)){
//             scoreBoard[nr][nc] |= adjTo(currentTurn);
//         }
//     }

//     Color clr = captureResultbyMove(r, c);
//     if(clr != EMPTY)
//         return clr;
    
//     switchTurn();
//     moveCount++;
//     return EMPTY;
// }

// Color Game::updateScoreAfter(Move move){
//     if(move == PASSMOVE)
//         return EMPTY;
        
//     if(moveCount >= 2)
//         updateScore(move.first, move.second);

//     if(getLegalMoveCount() == 0 || moveCount > boardSize){
//         return gameEnd();
//     }
//     return EMPTY;
// }

void Game::onGameEnd(Color winner){
    std::cout << "game over! winner is : " << static_cast<int>(winner) << std::endl;
}
#include "gamerules.h"

void Game::mergeChains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2) {
    uint8_t h1 = findHead(r1, c1), h2 = findHead(r2, c2);
    if (h1 == h2) return;
    
    if (chains[h1].size < chains[h2].size) std::swap(h1, h2);
    chains[h1].size += chains[h2].size;
    chains[h1].pseudoLibs += chains[h2].pseudoLibs;
    
    uint8_t cur = h2, start = h2;
    do {
        stones[cur / colSize][cur % colSize].head = h1;
        cur = stones[cur / colSize][cur % colSize].next;
    } while (cur != start);
    
    std::swap(stones[h2 / colSize][h2 % colSize].next, stones[h1 / colSize][h1 % colSize].next);
}


color Game::captureResultbyMove(uint8_t r, uint8_t c){
    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {cord, 1U, 0};
    
    for (size_t i = 0; i < 4; ++i) {
        uint8_t nr = r + dr[i], nc = c + dc[i];
        if (!inbound(nr, nc)) continue;

        if(board[nr][nc] == EMPTY) 
            ++(chains[findHead(r, c)].pseudoLibs);
        if (board[nr][nc] == board[r][c]){
            --(chains[findHead(nr, nc)].pseudoLibs);
            mergeChains(r, c, nr, nc);
        }
        else if (board[nr][nc] == reverseColor(board[r][c]) && --(chains[findHead(nr, nc)].pseudoLibs) == 0) 
            return board[r][c];
    }

    if(chains[findHead(r, c)].pseudoLibs == 0)
        return reverseColor(board[r][c]);
    
    return EMPTY;
}


uint8_t Game::checkScore(uint8_t r, uint8_t c, color clr) {
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

            for (size_t i = 0; i < 4; ++i) {
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

bool Game::canbeScore(uint8_t r, uint8_t c, color clr){
    if (r == 0U || c == 0U || r == rowSize-1 || c == colSize-1) return true;

    int cnt = 0;
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++)
                cnt += ((board[r+dr][c+dc] == clr || board[r+dr][c+dc] == NEUTRAL) ? 1 : 0);

    return cnt >= 3;
}

void Game::updateScore(uint8_t r, uint8_t c) { // major bottleneck
    color toCheck = board[r][c];
    if(!canbeScore(r, c, toCheck))
        return;

    for (size_t i = 0; i < 4; ++i) {
        uint8_t tr = r + dr[i], tc = c + dc[i];
        score[toCheck] += static_cast<float>(checkScore(tr, tc, toCheck));
    }
    finalScore = score[BLACK] - score[WHITE] - komi;
}

void Game::getScore(){
    std::queue<std::pair<uint8_t, uint8_t>> q;
    std::vector<std::pair<uint8_t, uint8_t>> emptyCells;

    for(size_t clr = 0; clr < 2; ++clr)
        for(size_t r = 0; r<rowSize; ++r)
            for(size_t c = 0; c<colSize; ++c){
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
                    scoreBoard[tr][tc] = static_cast<color>(clr);
                }
                score[clr] += areaCount;
            }
            
    finalScore = score[BLACK] - score[WHITE] - komi;
}

color Game::gameEnd(){
    return finalScore > 0 ? BLACK : WHITE;
}

uint8_t Game::getLegalMoveCount() const{
    uint8_t ret = 0;
    for(size_t i=0; i<rowSize; ++i)
        for(size_t j=0; j<colSize; ++j)
            ret += isLegal(i, j) ? 1 : 0;
    
    return ret;
}

color Game::makeMove(int r, int c){
    if(r < 0){ // resign
        switchTurn();
        return currentTurn;
    }

    if(r == rowSize){ // pass
        switchTurn();
        moveCount++;
        return EMPTY;
    }

    // update board & scoreBoard
    board[r][c] = currentTurn;
    scoreBoard[r][c] = NEUTRAL; // works as if neutral stone
    for(size_t i=0; i<4; ++i){ // make sure it can't be used for opponent
        uint8_t nr = r + dr[i];
        uint8_t nc = c + dc[i];
        if(inbound(nr, nc)){
            scoreBoard[nr][nc] |= adjTo(currentTurn);
        }
    }

    color clr = captureResultbyMove(r, c);
    if(clr != EMPTY)
        return clr;
    if(moveCount >= 2)
        updateScore(r, c);
    
    switchTurn();
    moveCount++;

    if(getLegalMoveCount() == 0 || moveCount > boardSize){
        return gameEnd();
    }
    return EMPTY;
}

void Game::onGameEnd(color winner){
    std::cout << "game over! winner is : " << winner << std::endl;
}

Game::Game(){
    for(size_t i=0; i<rowSize; ++i)
        for(size_t j=0; j<colSize; ++j){
            board[i][j] = EMPTY;
            scoreBoard[i][j] = EMPTY;
            mark[i][j] = 0;
        }
    
    currentTurn = BLACK;
    score[BLACK] = 0.0f;
    score[WHITE] = 0.0f;
    finalScore = 0.0f;
    moveCount = 0;
    visitId = 0;
    board[neutral.first][neutral.second] = NEUTRAL;
    scoreBoard[neutral.first][neutral.second] = NEUTRAL;
}

color Game::scoreWinner() const{
    return score[BLACK] - score[WHITE] - komi > 0 ? BLACK : WHITE;
}

color Game::getTurn() const{
    return currentTurn;
}

void Game::getBoardStatus() const{
    for(size_t i=0; i<rowSize; ++i)
        for(size_t j=0; j<colSize; ++j){
            std::cout << i << " " << j << " " << board[i][j] << " " << scoreBoard[i][j] << " ";
        }
}

void Game::displayBoardGUI(bool showScore) const{
    char display[rowSize][colSize];

    for(size_t i=0; i<rowSize; ++i){
        for(size_t j=0; j<colSize; ++j){
            switch(board[i][j]){
                case BLACK:
                    display[i][j] = 'o';
                    break;
                case WHITE:
                    display[i][j] = 'x';
                    break;
                case NEUTRAL:
                    display[i][j] = '+';
                    break;
                default:
                    display[i][j] = '-';
                    break;
            }

            if(showScore){
                switch(scoreBoard[i][j]){
                    case BLACK:
                        display[i][j] = 'b';
                        break;
                    case WHITE:
                        display[i][j] = 'w';
                        break;
                    default:
                        break;
                }
            }
        }
    }

    for(size_t i=0; i<rowSize; ++i){
        for(size_t j=0; j<colSize; ++j){
            std::cout << display[i][j] << " ";
        }
        std::cout << "\n";
    }

    // for(size_t i=0; i<rowSize; ++i){
    //     for(size_t j=0; j<colSize; ++j){
    //         std::cout << static_cast<int>(scoreBoard[i][j]) << " ";
    //     }
    //     std::cout << "\n";
    // }
}
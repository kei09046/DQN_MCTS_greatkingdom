#include "gamerules.h"
#include <utility>
#include <queue>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <bitset>
#include <cassert>


Game::Game() : board(globalConfig.neutrals), moveCount(0), possibleMoves(), threat(RESIGNMOVE), winmove(RESIGNMOVE), scoreChecker(){
    uint8_t temp = 0;
    for(int i=0; i<rowSize; ++i)
        for(int j=0; j<colSize; ++j){
            chains[temp] = {0, {}};
            stones[i][j] = {temp, temp};
            temp++;
        }
    
    currentTurn = BLACK;
    lastTwoMoves[0] = PASSMOVE;
    lastTwoMoves[1] = PASSMOVE;
}

// Game::Game(const Game& parent) : possibleMoves(), threat(RESIGNMOVE), winmove(RESIGNMOVE){
//     pTransferGroups.fill(BASEGROUPID);

//     currentTurn = parent.currentTurn;
//     for(int i=0; i<rowSize; ++i){
//         for(int j=0; j<colSize; ++j){
//             board[i][j] = parent.board[i][j];
//         }
//     }
//     for(int i=0; i<2; ++i){
//         lastTwoMoves[i] = parent.lastTwoMoves[i];
//         score[i] = parent.score[i];
//     }
//     moveCount = parent.moveCount;

//     for(int i=0; i<boardSize; ++i){
//         chains[i] = parent.chains[i];
//     }
//     for(int i=0; i<rowSize; ++i){
//         for(int j=0; j<colSize; ++j){
//             stones[i][j] = parent.stones[i][j];
//         }
//     }
// }

std::pair<Color, Wintype> Game::make_move(Move move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;
    if(move == RESIGNMOVE){ // resign
        switch_turn();
        return {currentTurn, RESIGN};
    }

    if(move == PASSMOVE){ // pass
        scoreChecker.update_move(board, move);
        switch_turn();
        moveCount++;
        return {EMPTY, NONE};
    }

    uint8_t r = move.first;
    uint8_t c = move.second;
    // if(r >= rowSize || c >= colSize) [[unlikely]]
    //     throw std::out_of_range("Game::makeMove : move off the board");

    board.place_stone(move, currentTurn);

    // update capture related functions
    Color clr = capture_result_after(r, c);
    if(clr != EMPTY)
        return {clr, CAPTURE};

    // update score related functions
    scoreChecker.update_move(board, {r, c});
    
    switch_turn();
    moveCount++;

    if(get_legal_move_count() == 0 || moveCount > boardSize){
        return {score_winner(), SCORE};
    }
    return {EMPTY, NONE};
}

Color Game::make_move_given_score(const Move& move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;

    if(move == PASSMOVE){ // pass
        scoreChecker.update_move_given_score(board, move);
        switch_turn();
        moveCount++;
        return EMPTY;
    }

    uint8_t r = move.first;
    uint8_t c = move.second;

    if(!board.is_empty(move)){
        print_move(move);
        display_board_gui(true);
    }
    board.place_stone(move, currentTurn);

    Color clr = capture_result_after(r, c);
    if(clr != EMPTY)
        return clr;

    // after place_stone (color is read from board) and capture check, same order as make_move.
    scoreChecker.update_move_given_score(board, move);

    switch_turn();
    moveCount++;

    if(get_legal_move_count() == 0 || moveCount > boardSize){
        return score_winner();
    }

    return EMPTY;
}

std::tuple<Color, Wintype, std::vector<float>> Game::make_move_with_stat(Move move){
    lastTwoMoves[0] = lastTwoMoves[1];
    lastTwoMoves[1] = move;

    assert(move != RESIGNMOVE);
    if(move == PASSMOVE){ // pass
        scoreChecker.update_move(board, move);
        switch_turn();
        moveCount++;
        return {EMPTY, NONE, {}};
    }

    uint8_t r = move.first;
    uint8_t c = move.second;

    if(!board.is_empty(move)){
        display_board_gui(true);
        print_move(move);
    }
    board.place_stone(move, currentTurn);

    // update capture related functions
    auto [clr, captured] = capture_result_with_stat(r, c);
    // update score related functions. Turn is already switched here, so use the color of the placed stone.
    scoreChecker.update_move(board, {r, c});

    switch_turn();
    moveCount++;

    // end by capture
    if(clr != EMPTY){
        std::vector<float> captureMap(boardSize, 0.0f);
        for(int i : captured)
            captureMap[i] = 1.0f;
        return {clr, CAPTURE, captureMap};
    }
    // end by score
    else if(get_legal_move_count() == 0 || moveCount > boardSize){
        std::vector<float> occupyMap(boardSize);
        const BitBoard bOccupy = board.stones(BLACK) | board.score(BLACK);
        const BitBoard wOccupy = board.stones(WHITE) | board.score(WHITE);
        for(int i=0; i<boardSize; ++i)
            occupyMap[i] = bOccupy[i] ? -1.0f : (wOccupy[i] ? 1.0f : 0.0f);
        return {score_winner(), SCORE, occupyMap};
    }
    return {EMPTY, NONE, {}};
}

void Game::set_policy_mask(){
    // first reset informations from parent which are no longer needed.
    possibleMoves.clear();
    winmove = RESIGNMOVE;
    threat = RESIGNMOVE;

    // update score map topology
    scoreChecker.update_topology(board);

    const auto& [tactic, wdl] = tactic_check();
    if(wdl > 0){
        winmove = tactic;
        possibleMoves.push_back(winmove.first * colSize + winmove.second);
        return;
    }
    else if(wdl < 0){
        threat = tactic;
        possibleMoves.push_back(PASSMOVE.first * colSize + PASSMOVE.second);
        return;
    }
    else{
        threat = tactic;
        if(threat != RESIGNMOVE){
            auto best = scoreChecker.maximize_score(board, currentTurn, threat);
            possibleMoves.push_back(best);
            return;
        }

        else{
            auto optimals = scoreChecker.maximize_score(board, currentTurn, 0U);
            for(const auto& m : optimals)
                possibleMoves.push_back(m);

            if(score_winner() == currentTurn)
                possibleMoves.push_back(PASSMOVE.first * colSize + PASSMOVE.second);
            return;
        }
    }
}

// return a list of cords which does not have superior score building move.
// only return moves that gains equal or more points than minScore.
std::vector<int> Game::maximize_score(const Color c, const uint8_t minScore) const{
    return scoreChecker.maximize_score(board, c, minScore);
}

// find a move that maximizes score which includes cord. If no move generates score, returns cord.
int Game::maximize_score(Color c, const Move& cord) const{
    return scoreChecker.maximize_score(board, c, cord);
}

std::vector<float> Game::transfer_logit(const std::vector<float>& logits) const{
    return scoreChecker.transfer_logit(logits, possibleMoves, currentTurn);
}

std::pair<Move, int> Game::tactic_check() const{
    Move tactic = RESIGNMOVE;

    std::bitset<boardSize> chainChecker;
    for(int i=0; i<rowSize; ++i){            
        for(int j=0; j<colSize; ++j){
            const Chain c = get_chain({i, j});
            if(c.size != 0 && c.liberties.count() == 1 && !chainChecker.test(get_chain_idx({i, j}))){
                chainChecker.set(get_chain_idx({i, j}), true);
                int onlyLib = c.liberties._Find_first();

                if(is_legal(onlyLib / colSize, onlyLib % colSize)){
                    // if my stone is under threat -> have to find only move unless can capture opponent's stone.
                    if(board.stones(currentTurn)[to_idx(i, j)]){
                        tactic = {onlyLib / colSize, onlyLib % colSize};
                    }

                    // if opponent stone is capturable
                    else{
                        return {{onlyLib / colSize, onlyLib % colSize}, 1};
                    }
                }
            }
        }
    }

    if(tactic == RESIGNMOVE)
        return {RESIGNMOVE, 0};

    // check if playing at threat would extend it's liberties
    std::bitset<boardSize> liberties;
    for(auto [r, c] : neighbors4(tactic)){
        if(board.is_empty(to_idx(r, c))){
            return {tactic, 0};
        }
        else if(board.stones(currentTurn)[to_idx(r, c)]){
            liberties |= chains[find_head(r, c)].liberties;
        }
    }
    liberties[tactic.first * colSize + tactic.second] = false;

    assert(tactic != RESIGNMOVE);
    if(liberties.none())
        return {tactic, -1};
    return {tactic, 0};
}


void Game::on_game_end(Color winner){
    std::cout << "game over! winner is : " << static_cast<int>(winner) << std::endl;
}


void Game::merge_chains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2) {
    //std::cout << "merging chain :" << (int)r1 << (int)c1 << " " << (int)r2 << (int)c2 << std::endl;
    uint8_t h1 = find_head(r1, c1), h2 = find_head(r2, c2);
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


Color Game::capture_result_after(uint8_t r, uint8_t c){
    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {1U, {}};
    Color sColor = get_board({r, c});

    for (auto [nr, nc] : neighbors4(r, c)) {
        Color adjColor = get_board({nr, nc});
        if(adjColor == EMPTY){ 
            chains[find_head(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (adjColor == sColor){
            merge_chains(r, c, nr, nc);
        }
        else if (adjColor == reverse_color(sColor)){ 
            auto& adjChain = chains[find_head(nr, nc)];
            adjChain.liberties.set(r * colSize + c, false);
            if(adjChain.liberties.none())
                return sColor;
        }
        // else adjacent to neutral; nothing should happen
    }

    if(chains[find_head(r, c)].liberties.none())
        return reverse_color(sColor);
    
    return EMPTY;
}

std::pair<Color, std::vector<int>> Game::capture_result_with_stat(uint8_t r, uint8_t c){
    Color winner = EMPTY;
    std::vector<uint8_t> capturedChainIdx;

    uint8_t cord = static_cast<uint8_t>(r * colSize + c);
    stones[r][c] = {cord, cord}; // head, next
    chains[r * colSize + c] = {1U, {}};
    Color sColor = get_board({r, c});

    for (auto [nr, nc] : neighbors4(r, c)) {
        Color adjColor = get_board({nr, nc});
        if(adjColor == EMPTY){ 
            chains[find_head(r, c)].liberties.set(nr * colSize + nc, true);
        }
        else if (adjColor == sColor){
            merge_chains(r, c, nr, nc);
        }
        else if (adjColor == reverse_color(sColor)){ 
            auto& adjChain = chains[find_head(nr, nc)];
            adjChain.liberties.set(r * colSize + c, false);
            if(adjChain.liberties.none()){
                winner = sColor;
                capturedChainIdx.push_back(find_head(nr, nc));
            }
        }
        // else adjacent to neutral; nothing should happen
    }

    if(winner == EMPTY && chains[find_head(r, c)].liberties.none()){
        winner = reverse_color(sColor);
        capturedChainIdx.push_back(find_head(r, c));
    }

    if(winner != EMPTY){
        std::vector<int> captured;

        for(auto idx : capturedChainIdx){
            int v = (winner == BLACK) ? 0 : 1;
            uint8_t repl = (winner == BLACK) ? BSCORE : WSCORE;

            //score[v] += chains[idx].size;

            uint8_t cur = idx, start = idx;
            do {
                captured.push_back(cur);
                uint8_t cr = cur / colSize;
                uint8_t cc = cur % colSize;
                // board[cr][cc] = repl;
                // for(int i=0; i<4; ++i){
                //     if(inbound(cr + dr[i], cc + dc[i]))
                //         chains[findHead(cr + dr[i], cc + dc[i])].liberties.set(cur, true);
                // }
                cur = stones[cr][cc].next;
            } while (cur != start);
        }
        return {winner, captured};
    }

    return {EMPTY, {}};
}


// std::vector<int> Game::candidates_when_threat(const std::bitset<boardSize>& potScore) const{
//     std::vector<int> candidates;
//     if(potScore.none()){
//         candidates = {threat.first * colSize + threat.second};
//         return candidates;
//     }
        
//     // find shortest path from T to opposite color stone.
//     // parent : BFS tree toward the closest opposite color stone. NOVISIT = not reached.
//     constexpr uint8_t NOVISIT = 255U;
//     std::array<uint8_t, boardSize> parent;
//     parent.fill(NOVISIT);
//     std::queue<Move> q;
//     Move oppStone = RESIGNMOVE;
//     Move option = RESIGNMOVE;

//     const int threatIdx = threat.first * colSize + threat.second;
//     parent[threatIdx] = threatIdx;
//     q.push(threat);
//     const BitBoard& adjToOpp = board.adjacent(reverse_color(currentTurn));
//     while(!q.empty()){
//         Move m = q.front();
//         q.pop();
//         const uint8_t mIdx = m.first * colSize + m.second;
//         for(auto [nr, nc] : neighbors4(m)){
//             const int nIdx = nr * colSize + nc;
//             if(!board.is_empty(nIdx))
//                 continue;
//             if(adjToOpp[nIdx]){
//                 q = {};
//                 oppStone = {nr, nc};
//                 parent[nIdx] = mIdx;
//                 break;
//             }
//             else if(parent[nIdx] == NOVISIT){
//                 q.push({nr, nc});
//                 parent[nIdx] = mIdx;
//             }
//         }
//     }

//     // extremely rare case where there are no reachable opposite color stone yet it is not territory.
//     if(oppStone == RESIGNMOVE){
//         for(uint8_t i=0U; i<rowSize; ++i){
//             for(uint8_t j=0U; j<colSize; ++j){
//                 if(parent[i * colSize + j] != NOVISIT && potScore[i * colSize + j])
//                     candidates.push_back(i * colSize + j);
//             }
//         }

//         // if threat is not inserted, insert threat. (threat is always visited)
//         if(!potScore[threatIdx]){
//             candidates.push_back(threatIdx);
//         }
//     }

//     // for each cord in path, should check if making a move there would make T a territory.
//     else{
//         option = oppStone;
//         while(option != threat){
//             if(potScore[option.first * colSize + option.second])
//                 candidates.push_back(option.first * colSize + option.second);

//             const auto p = parent[option.first * colSize + option.second];
//             assert(p != NOVISIT);
//             option = {p / colSize, p % colSize};
//         }
//         candidates.push_back(threatIdx);
//     }
//     return candidates;
// }

uint8_t Game::get_legal_move_count() const{
    return static_cast<uint8_t>(board.empty().count());
}

void Game::display_board_gui(bool showScore) const{
    char display[rowSize][colSize];

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            switch(get_board({i, j})){
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
                switch(get_score_board({i, j})){
                    case BSCORE:
                        display[i][j] = 'b';
                        break;
                    case WSCORE:
                        display[i][j] = 'w';
                        break;
                    default:
                        break;
                }
            }
        }
    }

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            std::cout << display[i][j] << " ";
        }
        std::cout << std::endl;
    }

	// const auto nnInput = PolicyValueNet::getData(game);
	// int cnt = rowSize * colSize * 18;
	// for(int i=18; i<=21; ++i){
	// 	std::cout << "\n channel " << i << "\n";
		
	// 	for(int j=0; j<rowSize; ++j){
	// 		for(int k=0; k<colSize; ++k)
	// 			std::cout << nnInput[cnt++] << " ";
	// 		std::cout << "\n";
	// 	}
	// }
}

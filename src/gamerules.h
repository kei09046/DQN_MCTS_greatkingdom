#ifndef GAMERULES_H
#define GAMERULES_H

#include <utility>
#include <queue>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <bitset>
#include "consts.h"


struct Chain {
    uint8_t size;        // Number of stones in the chain
    //std::unordered_set<int> liberties;
    std::bitset<boardSize> liberties;
};

struct Stone {
    uint8_t next;   // Next stone in circular linked list
    uint8_t head;   // Head of the chain
};


class Game{
public:
    Game();
    inline bool isLegal(uint8_t r, uint8_t c) const{
        return (board[r][c] == EMPTY) && (scoreBoard[r][c] & EMPTY);
    }

    std::bitset<outputSize> getLegalMoves() const{
        std::bitset<outputSize> legal;
        for(uint8_t i=0; i<rowSize; ++i){
            for(uint8_t j=0; j<colSize; ++j){
                legal[i * colSize + j] = isLegal(i, j);
            }
        }

        legal.set(rowSize * colSize);
        return legal;
    }

    void onGameEnd(color winner);

    color makeMoveNoScoreUpdate(Move move);

    color updateScoreAfter(Move move);

    color makeMove(Move move);

    inline float scoreDiff(color turn) const {
        return (score[BLACK] - score[WHITE] - globalConfig.komi) * ((turn == BLACK) ? 1.0f : -1.0f);
    };

    inline color scoreWinner() const {
        return score[BLACK] - score[WHITE] - globalConfig.komi > 0 ? BLACK : WHITE;
    };

    inline color getTurn() const{
        return currentTurn;
    };

    inline int getMoveCount() const{
        return moveCount;
    };

    inline color getBoard(u_int r, u_int c) const{
        return board[r][c];
    }

    inline color getScoreBoard(u_int r, u_int c) const{
        return scoreBoard[r][c];
    }

    inline const Chain& getChain(u_int idx) const{
        return chains[stones[idx/colSize][idx%colSize].head];
    }

    inline const Stone& getStone(u_int r, u_int c) const{
        return stones[r][c];
    }

    inline Move getLastMove(int idx) const{
        return lastTwoMoves[idx];
    }

    static inline color reverseColor(color c){
        return (c == BLACK) ? WHITE : BLACK;
    }

private:    
    color currentTurn;
    color board[rowSize][colSize];
    color scoreBoard[rowSize][colSize];
    Move lastTwoMoves[2];
    int visitId;
    int moveCount;
    float score[2];
    float finalScore;

    uint8_t mark[rowSize][colSize];

    Chain chains[boardSize];   // Chain data
    Stone stones[rowSize][colSize];    // Stone linked list info


    inline static bool inbound(int r, int c){
        return (r >= 0) && (r < rowSize) && (c >= 0) && (c < colSize);
    }

    inline static bool oppstate(color x, color y){
        return (x == BLACK && y == WHITE) || (x == WHITE && y == BLACK);
    }

    inline static uint8_t adjToOpposite(color clr){
        return (clr == BLACK) ? ADJTOWHITE : ADJTOBLACK;
    }

    inline static uint8_t adjTo(color clr){
        return (clr == BLACK) ? ADJTOBLACK : ADJTOWHITE;
    }

    inline void switchTurn(){
        currentTurn = reverseColor(currentTurn);
    }

    inline uint8_t findHead(int r, int c) { return stones[r][c].head; }

    void mergeChains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2);

    color captureResultbyMove(uint8_t r, uint8_t c);

    bool canbeScore(uint8_t r, uint8_t c, color clr);

    uint8_t checkScore(uint8_t r, uint8_t c, color clr);

    void getScore();

    void updateScore(uint8_t r, uint8_t c);

    color gameEnd();

    uint8_t getLegalMoveCount() const;
};

#endif
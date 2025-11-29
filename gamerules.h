#ifndef GAMERULES_H
#define GAMERULES_H

#include <utility>
#include <queue>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include "consts.h"


struct Chain {
    uint8_t head;        // Index of the head stone
    uint8_t size;        // Number of stones in the chain
    int pseudoLibs;  // Pseudoliberty count
};

struct Stone {
    uint8_t next;   // Next stone in circular linked list
    uint8_t head;   // Head of the chain
};


class Game{
private:    
    color currentTurn;
    color board[rowSize][colSize];
    color scoreBoard[rowSize][colSize];
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

    void updateScore(uint8_t r, uint8_t c);

    void getScore();

    color gameEnd();

    uint8_t getLegalMoveCount() const;

public:
    Game();
    inline bool isLegal(uint8_t r, uint8_t c) const{
        return (board[r][c] == EMPTY) && (scoreBoard[r][c] & EMPTY);
    }

    void onGameEnd(color winner);

    color makeMove(int r, int c);

    color scoreWinner() const;

    color getTurn() const;

    void displayBoardGUI(bool showScore = true) const;

    inline color getBoard(u_int r, u_int c) const{
        return board[r][c];
    }

    inline color getScoreBoard(u_int r, u_int c) const{
        return scoreBoard[r][c];
    }

    static inline color reverseColor(color c){
        return (c == BLACK) ? WHITE : BLACK;
    }
};

#endif
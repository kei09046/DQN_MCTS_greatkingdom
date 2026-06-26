#ifndef GAMERULES_H
#define GAMERULES_H

#include <utility>
#include <queue>
#include <vector>
#include <bitset>
#include "consts.h"


struct Chain {
    uint8_t size;        // Number of stones in the chain
    std::bitset<boardSize> liberties;
};

struct Stone {
    uint8_t next;   // Next stone in circular linked list
    uint8_t head;   // Head of the chain
};

struct SegInfo{
    std::vector<uint8_t> connectedSeg;
    bool adjOpp;
    std::bitset<4> adjEdge;
};


class Game{
public:
    Game();
    inline bool isLegal(uint8_t r, uint8_t c) const{
        return board[r][c] & EMPTY;
    }

    inline bool isLegal(Move m) const{ // return true if square is not occupied / nor is score of any side.
        return (m == RESIGNMOVE) || isLegal(m.first, m.second);
    }

    std::bitset<outputSize> getLegalMoves() const{
        std::bitset<outputSize> legal;
        for(uint8_t i=0; i<rowSize; ++i){
            for(uint8_t j=0; j<colSize; ++j){
                legal[i * colSize + j] = isLegal(i, j);
            }
        }

        legal[rowSize * colSize] = true;
        return legal;
    }

    void onGameEnd(Color winner);

    std::pair<Color, Wintype> makeMove(Move move);

    // unlike makeMove function, do not return immediately even if terminal condition is met. Check every detail.
    std::tuple<Color, Wintype, std::vector<float>> makeMoveWithStat(Move move);

    std::pair<Move, int> threatCheck() const;

    // return type : winMove(resignMove if nothing), list of moves, list of nextGames, transferTable. 
    std::tuple<std::pair<Move, int>, std::vector<Move>, std::vector<Game>, std::vector<std::vector<uint8_t>>> expand(const Move threat) const;

    inline float scoreDiff(Color turn) const { // does not calculate komi; Just return raw difference in territory.
        return (score[0] - score[1]) * ((turn == BLACK) ? 1.0f : -1.0f);
    };

    inline Color scoreWinner() const {
        return score[0] - score[1] - globalConfig.komi > 0 ? BLACK : WHITE;
    };

    inline Color getTurn() const{
        return currentTurn;
    };

    inline int getMoveCount() const{
        return moveCount;
    };

    inline Color getBoard(Move m) const{
        return board[m.first][m.second] & BOARDMASK;
    }

    inline Color getScoreBoard(Move m) const{
        return board[m.first][m.second] & SCOREMASK;
    }

    inline const int getChainIdx(int cord) const{
        return stones[cord/colSize][cord%colSize].head;
    }

    inline const Chain& getChain(int cord) const{
        return chains[stones[cord/colSize][cord%colSize].head];
    }

    inline const Chain& getChain(Move m) const{
        return chains[stones[m.first][m.second].head];
    }

    inline const Stone& getStone(Move m) const{
        return stones[m.first][m.second];
    }

    inline Move getLastMove(int idx) const{
        return lastTwoMoves[idx];
    }

    static inline Color reverseColor(Color c){
        return (c == BLACK) ? WHITE : BLACK;
    }

private:    
    Color currentTurn;
    Color board[rowSize][colSize];
    Move lastTwoMoves[2];
    int moveCount;
    float score[2];

    Chain chains[boardSize];   // Chain data
    Stone stones[rowSize][colSize];    // Stone linked list info


    // overflow is managed.
    inline static bool inbound(uint8_t r, uint8_t c){
        return (r < rowSize) && (c < colSize);
    }

    inline static bool oppstate(Color x, Color y){
        return (x == BLACK && y == WHITE) || (x == WHITE && y == BLACK);
    }

    inline static uint8_t adjToOpposite(Color clr){
        return (clr == BLACK) ? ADJTOWHITE : ADJTOBLACK;
    }

    inline static uint8_t adjTo(Color clr){
        return (clr == BLACK) ? ADJTOBLACK : ADJTOWHITE;
    }

    inline void switchTurn(){
        currentTurn = reverseColor(currentTurn);
    }

    // capture related functions
    inline uint8_t findHead(int r, int c) const { return stones[r][c].head; }

    void mergeChains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2);

    Color captureResultbyMove(uint8_t r, uint8_t c);

    std::pair<Color, std::vector<float>> captureResultWithStat(uint8_t r, uint8_t c);

    // score related functions
    void updateScore(uint8_t r, uint8_t c);

    bool canbeScore(uint8_t r, uint8_t c, Color clr) const;

    // calculates if territory is formed on given grid using BFS.
    // Also updates game state.
    uint8_t checkScore(uint8_t r, uint8_t c, Color clr);

    /////////////////////////////////////////////////////////
    // Functions needed for effective expansion.

    // Faster version of checkScore but segTable should be provided.
    // Calculate whether playing at {r, c} would give territory, without changing game state.
    // Returns bitset with 1 if corresponding segment becomes territory.
    std::bitset<boardSize> checkScore(uint8_t r, uint8_t c, Color clr, const std::pair<std::vector<SegInfo>, std::array<uint8_t, boardSize>>& segTable) const;

    std::pair<std::vector<SegInfo>, std::array<uint8_t, boardSize>> segmentTable(const std::bitset<boardSize>& potScore) const;

    std::vector<Move> possibleMovesWhenThreat(const Move& threat, const std::bitset<boardSize>& potScore) const;

    // first value of acquired is ignored to use transferTable as an argument.
    Color makeMoveNoScoreUpdate(const Move& move, const std::vector<uint8_t>& acquired);

    /////////////////////////////////////////////////////////////////////////

    uint8_t getLegalMoveCount() const;
};

#endif
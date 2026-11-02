#ifndef GAMERULES_H
#define GAMERULES_H

#include <utility>
#include <queue>
#include <vector>
#include <bitset>
#include "consts.h"

using SegId = uint8_t;
using GroupId = uint8_t;

struct Chain {
    uint8_t size;        // Number of stones in the chain
    std::bitset<boardSize> liberties;
};

struct Stone {
    uint8_t next;   // Next stone in circular linked list
    uint8_t head;   // Head of the chain
};

struct SegInfo{
    std::vector<SegId> connectedSeg;
    bool adjOpp;
    std::bitset<4> adjEdge;
};

struct SegTable{
    std::vector<SegInfo> segInformation;
    std::array<SegId, boardSize> segId;
};

enum Wintype{
    SCORE,
    CAPTURE,
    RESIGN,
    NONE
};


class Game{
public:
    Game();
    // copy constructor which gets called during MCTS search to generate child node.
    // do not copy transferGroups/availableMoves.
    Game(const Game& parent);

    inline bool is_legal(uint8_t r, uint8_t c) const{
        return board[r][c] & EMPTY;
    }

    inline bool is_legal(Move m) const{ // return true if square is not occupied / nor is score of any side.
        return (m == RESIGNMOVE) || is_legal(m.first, m.second);
    }

    std::bitset<outputSize> get_legal_moves() const{
        std::bitset<outputSize> legal;
        for(uint8_t i=0; i<rowSize; ++i){
            for(uint8_t j=0; j<colSize; ++j){
                legal[i * colSize + j] = is_legal(i, j);
            }
        }

        legal[rowSize * colSize] = true;
        return legal;
    }

    uint8_t get_legal_move_count() const;

    void on_game_end(Color winner);

    std::pair<Color, Wintype> make_move(Move move);

    // optimized version of makeMove. Used inside MCTS search only. setPolicyMask() should be called beforehand.
    Color make_move_given_score(const Move& move);

    // unlike makeMove function, do not return immediately even if terminal condition is met. Check every detail.
    // At game end also returns a map : the captured stones (1) for a capture, the occupancy (black -1 / white +1) for a score ending.
    std::tuple<Color, Wintype, std::vector<float>> make_move_with_stat(Move move);

    // set up list of possible moves, logit transfer list.
    // must be called before NN eval. 
    // During MCTS, called on first visit on that node.
    void set_policy_mask();

    std::pair<Move, int> tactic_check() const;

    std::vector<int> opponent_scoring_moves() const;

    // print the board to stdout. If showScore is set, territory is shown as 'b' / 'w'.
    void display_board_gui(bool showScore) const;

    inline float score_diff(Color turn) const { // does not calculate komi; Just return raw difference in territory.
        return (score[0] - score[1]) * ((turn == BLACK) ? 1.0f : -1.0f);
    };

    inline Color score_winner() const {
        return score[0] - score[1] - globalConfig.komi > 0 ? BLACK : WHITE;
    };

    inline Color get_turn() const{
        return currentTurn;
    };

    inline int get_move_count() const{
        return moveCount;
    };

    inline Color get_board(Move m) const{
        return board[m.first][m.second] & BOARDMASK;
    }

    inline Color get_score_board(Move m) const{
        return board[m.first][m.second] & SCOREMASK;
    }

    inline const int get_chain_idx(int cord) const{
        return stones[cord/colSize][cord%colSize].head;
    }

    inline const int get_chain_idx(Move m) const{
        return stones[m.first][m.second].head;
    }

    inline const Chain& get_chain(int cord) const{
        return chains[stones[cord/colSize][cord%colSize].head];
    }

    inline const Chain& get_chain(Move m) const{
        return chains[stones[m.first][m.second].head];
    }

    inline const Stone& get_stone(Move m) const{
        return stones[m.first][m.second];
    }

    inline Move get_last_move(int idx) const{
        return lastTwoMoves[idx];
    }

    inline const std::vector<int>& get_available_moves() const{
        return possibleMoves;
    }

    inline const std::array<GroupId, boardSize + 1>& get_transfer_table() const{
        return pTransferGroups;
    }

    inline Move get_threat() const{
        return threat;
    }

    inline Move get_win() const{
        return winmove;
    }

    static inline Color reverse_color(Color c){
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

    // for NN input.
    Move threat;
    Move winmove;
    std::vector<int> possibleMoves;
    std::array<GroupId, boardSize + 1> pTransferGroups;


    // overflow is managed.
    inline static bool inbound(uint8_t r, uint8_t c){
        return (r < rowSize) && (c < colSize);
    }

    inline static bool oppstate(Color x, Color y){
        return (x == BLACK && y == WHITE) || (x == WHITE && y == BLACK);
    }

    inline static uint8_t adj_to_opposite(Color clr){
        return (clr == BLACK) ? ADJTOWHITE : ADJTOBLACK;
    }

    inline static uint8_t adj_to(Color clr){
        return (clr == BLACK) ? ADJTOBLACK : ADJTOWHITE;
    }

    inline void switch_turn(){
        currentTurn = reverse_color(currentTurn);
    }

    // capture related functions
    inline uint8_t find_head(int r, int c) const { return stones[r][c].head; }

    void merge_chains(uint8_t r1, uint8_t c1, uint8_t r2, uint8_t c2);

    Color capture_result_after(uint8_t r, uint8_t c);

    std::pair<Color, std::vector<int>> capture_result_with_stat(uint8_t r, uint8_t c);

    // score related functions
    void update_score(uint8_t r, uint8_t c);

    bool canbe_score(uint8_t r, uint8_t c, Color clr) const;

    // calculates if territory is formed on given grid using BFS.
    // Also updates game state.
    uint8_t check_score(uint8_t r, uint8_t c, Color clr);

    /////////////////////////////////////////////////////////
    // Functions needed for effective expansion.

    // Faster version of checkScore but segTable should be provided.
    // Calculate whether playing at {r, c} would give territory, without changing game state.
    // Returns bitset with 1 if corresponding segment becomes territory.
    std::bitset<boardSize> check_score(uint8_t r, uint8_t c, Color clr, const SegTable& segTable) const;

    SegTable segment_table(const std::bitset<boardSize>& potScore, Color turn) const;

    // returns possible moves that handles imminent threat. Later options are inferior ones; If first candidate makes territory, no need to check any other options.
    std::vector<int> candidates_when_threat(const std::bitset<boardSize>& potScore) const;
    /////////////////////////////////////////////////////////////////////////
};

#endif
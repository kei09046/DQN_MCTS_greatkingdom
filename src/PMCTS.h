#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"
#include "evaluator.h"
#include "gamerules.h"
#include <vector>
#include <utility>
#include <unordered_map>

class Node;
using TransTable = std::unordered_map<HashValue, std::pair<Node*, unsigned int>>;

class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, TransTable* const transposTable);

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    void addDirichletNoise(Evaluator* evaluator);

    void deleteTree();

    void deleteTree(Node* exception);

    inline const std::vector<std::vector<uint8_t>>& TransferTable_() const{
        return transferTable;
    }

    inline const std::vector<Move>& availableMoves_() const{
        return availableMoves;
    }

private:
    friend class MCTS;

    const Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, initQ, S, Wp; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    // S : mean score difference Wp : mean win probability
    std::vector<float> edgeP;
    std::vector<float> edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.
    std::vector<std::vector<uint8_t>> transferTable; 
    const Color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation

    std::vector<Node*> child;
    std::vector<Move> availableMoves; // among game.isLegal() moves, consider actually useful moves.
    Move winmove; // if RESIGNMOVE : there is no winning move.
    int forcedState; // if 0 : not forced win/loss +k : win in k move -k : lose in k move

    std::shared_ptr<PolicyValueOutput> evaluation; // stores evaluation for this node. Used to temporarily hold evaluation.
    TransTable* const transposTable;

    void addChild(const Move& move, const Game& ng);

    void expand();

    int selectChildInSearch();
};


class alignas(64) MCTS{
public:
    MCTS(Evaluator* evaluator);
    ~MCTS();
    MCTS(MCTS&& other) noexcept;

    void runSimulation(const int playMode, const int nPlayout, const int timeLimit);

    Move getMove(float temperature);

    MoveData getMoveProb(float temperature);

    float getEval();

    void printVariation();

    bool jump(Move move);

    void reset();

    #ifdef measureTime
    std::vector<int> getTimeStats() const;
    
    void resetTimeStats();
    #endif

private:
    Node* root;
    Evaluator* evaluator; // shared along multiple MCTS instances
    TransTable* transposTable;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation, 
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int nPlayout, const int timeLimit);

    void updateEval(const std::shared_ptr<NNResultBuf> buf, const std::vector<Node*> path, Node* cur);

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW=0.0f, float evalS=0.0f);
};
#endif
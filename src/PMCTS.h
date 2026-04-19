#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"
#include "evaluator.h"
#include "gamerules.h"
#include <vector>
#include <utility>
#include <unordered_map>

std::vector<float> softmax(const std::vector<float>& logit, const std::vector<Move>& availableMoves);

std::vector<float> softmax(const std::vector<float>& logit);

std::pair<float, float> calculateQ(const std::vector<float>& winLogit, const std::vector<float>& scoreDist, float scoreShift,
    float score_factor = 0.03f,   // convert points to utility
    float risk_aversion = 0.003f    // penalty per standard deviation);
);

class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table);

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    void addDirichletNoise(Evaluator* evaluator);

    void deleteTree();

    void deleteTree(Node* exception);

private:
    friend class MCTS;

    const Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, initQ, S, Wp; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    // S : mean score difference Wp : mean win probability
    std::vector<float> edgeP;
    std::vector<float> edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.
    const Color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation

    std::vector<Node*> child;
    std::vector<Move> available_moves; // among game.isLegal() moves, consider actually useful moves.
    Move onlyMove; // if RESIGNMOVE : there is no only move. Set to other value if definite best move is found. 
    // Best move : forced winning move, forced move not to lose, Only move that gives chances in lost position.
    int forcedState; // if 0 : not forced win/loss +k : win in k move -k : lose in k move
    int losingMoveCount;
    std::unordered_map<HashValue, Node*>* const trans_table;

    void addChild(const int r, const int c, const Game& ng);

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
    std::unordered_map<HashValue, Node*>* trans_table;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation, 
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int nPlayout, const int timeLimit);

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW=0.0f, float evalS=0.0f);

    void propagate(const std::vector<Node*>& path, const std::vector<int>& childIdx, int forced);
};
#endif
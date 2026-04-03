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

std::pair<float, float> calculateQ(const std::vector<float>& winLogit, const std::vector<float>& scoreDist, float komi = globalConfig.komi,   // board komi
    float score_factor = 0.02f,   // convert points to utility
    float risk_aversion = 0.0f    // penalty per standard deviation);
);

class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table);

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    void addDirichletNoise(Evaluator* evaluator);

    #ifndef transTable
    void deleteTree();

    void deleteTree(Node* exception);
    #endif

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
    Move winmove;
    std::unordered_map<HashValue, Node*>* const trans_table;

    void addChild(const int r, const int c, const Game& ng);

    void expand();

    int selectChildInSearch();
};


class alignas(64) MCTS{
public:
    MCTS(Evaluator* evaluator, std::string mode="playout", int playout=400, int time=10);
    ~MCTS();
    MCTS(MCTS&& other) noexcept;

    void runSimulation();

    Move getMove(float temperature);

    MoveData getMoveProb(float temperature);

    bool jump(Move move);

    void reset();

    #ifdef measureTime
    std::vector<int> getTimeStats() const;
    
    void resetTimeStats();
    #endif

private:
    Node* root;
    int playMode, nPlayout, timeLimit;
    Evaluator* evaluator; // shared along multiple MCTS instances
    std::unordered_map<HashValue, Node*>* trans_table;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation, 
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck);

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW, float evalS);
};
#endif
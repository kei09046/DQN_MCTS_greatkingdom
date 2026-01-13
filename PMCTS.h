#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"

#ifdef apvMCTS
#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "hash.h"
#include "evaluator.h"
#include "dirichlet.h"
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <memory>
#include <atomic>
#include <latch>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>


enum NodeState_{
    NEEDEXPAND,
    EXPANDING,
    NEEDEVAL,
    EVALUATING,
    FINAL
};

class Node;
using NodeState = std::atomic<NodeState_>;
using TransTable = Cache<Node>;
constexpr static float errorReturn = 10.0f;

class alignas(64) Node{
private:
    const Game game; // includes position, territory, valid moves etc. for heuristic
    std::atomic<float> N, W, initQ; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    std::vector<std::atomic<float>> edgeP;
    std::vector<std::atomic<float>> edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.

    std::vector<Node*> child;
    std::vector<Move> available_moves; // among game.isLegal() moves, consider actually useful moves.
    Move winmove;
    const color turn;

    TransTable* const trans_table;
    Evaluator* evaluator;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation

    NodeState state;

    void addChild(int r, int c, Game ng);

    void expand();

    float evaluate();

    static std::vector<std::atomic<float>> softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves);

public:
    Node(const Game& g, const HashValue hashValue, TransTable* const trans_table, Evaluator* evaluator);

    float searchandPropagate();

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    #ifdef dirichletNoise
    void addDirichletNoise();
    #endif

    #ifndef transTable
    void deleteTree();

    void deleteTree(Node* exception);
    #endif
};

class alignas(64) MCTS{
private:
    Node* root;
    int playout;
    Evaluator* evaluator; // shared along multiple MCTS instances
    TransTable* trans_table;
    boost::asio::thread_pool thread_pool;

public:
    MCTS(int playout, Evaluator* evaluator);
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
};
#endif


#ifndef apvMCTS
#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "hash.h"
#include "evaluator.h"
#include "dirichlet.h"
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <unordered_map>
#include <memory>


std::vector<float> softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves);


class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table);

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    #ifdef dirichletNoise
    void addDirichletNoise(Evaluator* evaluator);
    #endif

    #ifndef transTable
    void deleteTree();

    void deleteTree(Node* exception);
    #endif

private:
    friend class MCTS;

    const Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, initQ; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    std::vector<float> edgeP;
    std::vector<float> edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.
    const color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation

    std::vector<Node*> child;
    std::vector<Move> available_moves; // among game.isLegal() moves, consider actually useful moves.
    Move winmove;
    std::unordered_map<HashValue, Node*>* const trans_table;

    void addChild(int r, int c, Game ng);

    void expand();

    int selectChildInSearch();
};


class alignas(64) MCTS{
public:
    MCTS(int nPlayout, Evaluator* evaluator);
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
    int nPlayout;
    Evaluator* evaluator; // shared along multiple MCTS instances
    std::unordered_map<HashValue, Node*>* trans_table;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation, 
        std::vector<std::vector<Node*>>& updateQueue, std::vector<NNResultBuf*>& resultBuffer, bool& searchStuck);
};
#endif

#endif
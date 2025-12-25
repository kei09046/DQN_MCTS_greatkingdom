#ifndef PMCTS_H
#define PMCTS_H

#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "memorypool.h"
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
#include <atomic>


class alignas(64) Node{
private:
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

    static std::vector<float> softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves);

public:
    Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table);

    float searchandPropagate(Evaluator* evaluator);

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
    std::unordered_map<HashValue, Node*>* trans_table;

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
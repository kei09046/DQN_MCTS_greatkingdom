#ifndef PMCTS_H
#define PMCTS_H

#include "gamerules.h"
#include "policyvalue.h"
#include "random.h"
#include "memorypool.h"
#include "hash.h"
#include "evalcache.h"
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <unordered_map>
#include <memory>


using MoveData = std::tuple<std::pair<int, int>, std::array<float, outputSize> >; // move + move possibility

class alignas(64) Node{
private:
    const Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, P, initQ; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    const color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation

    std::vector<Node*> child;
    std::vector<std::pair<int, int> > available_moves; // among game.isLegal() moves, consider actually useful moves.
    std::pair<int, int> winmove;
    EvalCache* const eval_cache;
    std::unordered_map<HashValue, Node*>* const trans_table;

    void expand();

    static std::vector<float> softmax(const std::vector<float>& logit, const std::vector<std::pair<int, int>>& available_moves);

public:
    Node(const Game& g, const HashValue hashValue, EvalCache* const eval_cache, std::unordered_map<HashValue, Node*>* const trans_table);

    float searchandPropagate(PolicyValueNet& net);

    std::pair<int, int> selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(std::pair<int, int> move);
};

class alignas(64) MCTS{
private:
    Node* root;
    int playout;
    PolicyValueNet* net;
    EvalCache* const eval_cache;
    std::unordered_map<HashValue, Node*>* const trans_table;

public:
    MCTS(int playout, PolicyValueNet* net, EvalCache* const eval_cache, std::unordered_map<HashValue, Node*>* const trans_table);

    void runSimulation();

    std::pair<int, int> getMove(float temperature);

    MoveData getMoveProb(float temperature);

    bool jump(std::pair<int, int> move);

    void reset();

    void updateModel(); // what to do when model gets updated. 

    #ifdef measureTime
    std::vector<int> getTimeStats() const;
    
    void resetTimeStats();
    #endif
};

#endif
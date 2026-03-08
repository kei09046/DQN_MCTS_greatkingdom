#ifndef MCTS_H
#define MCTS_H

#include "gamerules.h"
#include <vector>
#include <utility>
#include <cmath>

class Node{
private:
    const Game* game; // includes position, territory, valid moves etc. for heuristic
    float initEval;
    Color turn; 

    int winCount, searchCount;
    std::vector<float> searchPref;
    std::vector<Node*> child;
    std::vector<std::pair<int, int> > legal;
    std::pair<int, int> winmove;

    void expand();

    void deletetree();

public:
    Node(Game* g);

    ~Node();

    int searchandPropagate();

    std::pair<int, int> selectMove();

    Node* jump(std::pair<int, int> move);
};

class MCTS{
private:
    Node* root;

public:
    MCTS();
    
    MCTS(Node* root);

    void runSimulation(int simulCount);

    std::pair<int, int> getMove();

    void jump(std::pair<int, int> move);
};

#endif
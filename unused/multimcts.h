#ifndef MULTIMCTS_H
#define MULTIMCTS_H

#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "threadpool.h"
#include <vector>
#include <utility>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>

using MoveData = std::tuple<std::pair<int, int>, std::array<float, outputSize> >; // move + move possibility

const float cPuct = 0.1f;
const float decay = 0.9f;
const float virtualLoss = 1.0f;
const int searchThreadN = 16;
const int evaluateThreadN = 2;
const int minibatchSize = 8;

struct EvalRequest {
    const Game* game;
    std::vector<std::pair<int, int>>* legal;
    std::promise<PolicyValueOutput> promise;
};

class Evaluator{
private:
    //std::queue<EvalRequest> evalQueue;
    rigtorp::MPMCQueue<EvalRequest> evalQueue;
    std::condition_variable evalCondVar;
    std::mutex evalMutex; // activate eval thread by condition
    std::thread worker[evaluateThreadN];
    bool stopWorker;

    PolicyValueNet& net;

    void evalWorker();

public:
    std::atomic<int> activeSearchThread{0};

    Evaluator(PolicyValueNet& net);

    ~Evaluator();

    std::future<PolicyValueOutput> 
    enqueueEvaluation(const Game* game, std::vector<std::pair<int, int>>* legal);
};


class Node{
private:
    const Game* game; // includes position, territory, valid moves etc. for heuristic
    std::atomic<int> N{0};
    std::atomic<float> W{0};
    float P, initQ;
    Color turn;

    std::vector<Node*> child;
    std::vector<std::pair<int, int> > legal;
    std::pair<int, int> winmove;

    std::mutex expandMutex, evaluationMutex;
    std::atomic<bool> expanded{false}, evaluated{false}, evaluating{false};
    void expand();

    static std::vector<float> softmax(std::vector<float>& logit);

public:
    Node(Game* g);

    ~Node();

    void deletetree();

    float searchandPropagate(Evaluator& evaluator);

    std::pair<int, int> selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(std::pair<int, int> move);
};
    

class MCTS{
private:
    int playout;
    Node* root;
    PolicyValueNet net;
    Evaluator evaluator;
    ThreadPool searchPool; 

public:
    MCTS(int playout, PolicyValueNet& net);

    void runSimulation();

    std::pair<int, int> getMove(float temperature);

    MoveData getMoveProb(float temperature);

    void jump(std::pair<int, int> move);

    void reset();
};


#endif
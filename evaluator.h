// used to manage evaluation of multiple threads
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include "consts.h"
#include "evalcache.h"
#include "neuralNet.h"

struct NNResultBuf{
    std::condition_variable resultcv;
    std::mutex resultmutex;
    std::shared_ptr<PolicyValueOutput> result = nullptr;
};

struct evalRequest {
    NNResultBuf* buf;
    const Game* game;
};

class Evaluator{
private:
    PolicyValueNet* net;
    EvalCache cache;
    std::thread handler;
    std::mutex qmutex;
    std::condition_variable qcv;
    std::queue<evalRequest> q;
    bool stop;

    void createHandlerThreads();
    void HandlerWork(); // should collect evaluate requests and create vector of game, then call net->batchEvaluate

public:
    Evaluator(PolicyValueNet* net);
    Evaluator(const std::string& model_file, const std::string& model_type, bool use_gpu);
    Evaluator(const std::string& model_file, bool use_gpu);
    ~Evaluator();
    bool evaluate(NNResultBuf& buf, const Game* game, HashValue hash); // return true if cache hit
    void updateModel(PolicyValueNet* updatedNet);
};
#endif
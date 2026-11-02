// used to manage evaluation of multiple threads
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <coroutine>
#include "consts.h"
#include "evalcache.h"
#include "neuralNet.h"

struct NodeInfo{
    // policy
    std::vector<float> P;
    // raw value head output
    float V;
    // utility computed
    float U;
    // score difference prediction(without komi)
    float S;
};

using EvalCache = Cache<NodeInfo>;

struct NNResultBuf{
    std::condition_variable resultcv;
    std::mutex resultmutex;
    std::shared_ptr<NodeInfo> result = nullptr;
};

struct EvalRequest {
    std::shared_ptr<NNResultBuf> buf;
    const Game* game;
    HashValue hash;
    std::unique_ptr<NNInput> input;
};

class Evaluator{
private:
    PolicyValueNet* net;
    EvalCache cache;
    std::thread handler;
    std::mutex qmutex;
    std::condition_variable qcv;
    std::queue<EvalRequest> q;
    std::atomic<bool> stop;
    PinnedBuffers pinned; // used only by the handler thread

    // cache lookup; on a miss, builds the input in the calling thread and queues the request.
    bool request(const std::shared_ptr<NNResultBuf>& buf, const Game* game, HashValue hash);

    void create_handler_threads();
    void handler_work(); // collects evaluate requests, merges their inputs into one batch and calls net->batchEvaluate
    // convert raw NN output into values for MCTS search.
    NodeInfo convert_form(const PolicyValueOutput& nnOutput, const Game& game);
    void free_pinned_memory(); // called by the handler thread on its way out

public:
    Evaluator(PolicyValueNet* net);
    Evaluator(const std::string& modelFile, const std::string& modelType, int numBlocks, bool useGpu);
    Evaluator(const std::string& modelFile, bool useGpu);
    ~Evaluator();
    bool evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash); // return true if cache hit
    bool async_evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash);
    void update_model(PolicyValueNet* updatedNet);
    PolicyValueNet* model() const { return net; }
};
#endif
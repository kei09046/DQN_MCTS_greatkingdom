// used to manage evaluation of multiple threads
#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <atomic>
#include <condition_variable>
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
    const bool netOwned; // if set true, net should be deleted on destructor.
    EvalCache cache;
    std::thread handler;
    std::mutex qmutex;
    std::condition_variable qcv;
    std::queue<EvalRequest> q;
    std::atomic<bool> stop;
    PinnedBuffers pinned; // used only by the handler thread
    std::mutex netMutex; // held while the handler evaluates and while update_model changes the net

    // cache lookup; on a miss, builds the input in the calling thread and queues the request.
    bool request(const std::shared_ptr<NNResultBuf>& buf, const Game* game, HashValue hash);

    void create_handler_threads();
    void handler_work(); // collects evaluate requests, merges their inputs into one batch and calls net->batchEvaluate
    // convert raw NN output into values for MCTS search.
    NodeInfo convert_form(const PolicyValueOutput& nnOutput, const Game& game);
    void free_pinned_memory(); // called by the handler thread on its way out

public:
    // evaluates with net (not owned).
    Evaluator(PolicyValueNet* net, const CacheConfig& cache);
    // loads modelFile; newNet : the network built if the file name doesn't give type / size (see PolicyValueNet).
    Evaluator(const std::string& modelFile, const NetConfig& newNet, const CacheConfig& cache, bool useGpu);
    ~Evaluator();
    bool evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash); // return true if cache hit
    bool async_evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash);
    void update_model(PolicyValueNet* updatedNet);
    PolicyValueNet* model() const { return net; }
};
#endif
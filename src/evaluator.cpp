#include "evaluator.h"

void Evaluator::createHandlerThreads() {
    handler = std::thread(&Evaluator::HandlerWork, this);
}

void Evaluator::HandlerWork() {
    std::vector<const Game*> batchGames;
    std::vector<NNResultBuf*> batchBufs;
    std::vector<HashValue> hashBufs;

    while (!stop) {
        // wait until at least 1 request exists
        std::unique_lock<std::mutex> lk(qmutex);
        qcv.wait(lk, [&]{ return stop || !q.empty(); });

        // drain queue into batch
        while (!q.empty()) {
            auto r = q.front(); q.pop();
            batchBufs.push_back(r.buf);
            hashBufs.push_back(r.hash);
            batchGames.push_back(r.game);
        }
        lk.unlock();

        // run batch
        std::vector<PolicyValueOutput> outputs;
        // std::cerr << "attempting batch evaluation" << std::endl;
        while(true){
            try{
                outputs = net->batchEvaluate(batchGames);
                break;
            }catch(const c10::Error& e){
                std::cerr << "batch evaluate failed" << std::endl;
            }catch(const std::exception& e){
                std::cerr << e.what() << std::endl;
            }catch(...){
                std::cerr << "other exception occured" << std::endl;
            }
        }
        // std::cerr << "Batch evaluated successfully" << std::endl;

        // write results back & notify
        for (int i = 0; i < batchBufs.size(); i++) {
            auto buf = batchBufs[i];
            {
                std::lock_guard<std::mutex> lk2(buf->resultmutex);
                buf->result = std::make_shared<PolicyValueOutput>(outputs[i]);
                cache.insert(hashBufs[i], buf->result);
                //buf->result = std::make_shared<PolicyValueOutput>(outputs[i]);
            }
            buf->resultcv.notify_one();
        }

        batchGames.clear();
        batchBufs.clear();
    }
}

Evaluator::Evaluator(PolicyValueNet* net) : net(net), stop(false){
    createHandlerThreads();
}
Evaluator::Evaluator(const std::string& model_file, const std::string& model_type, bool use_gpu)
    : net(new PolicyValueNet(model_file, model_type, use_gpu)), stop(false){
    createHandlerThreads();
}
Evaluator::Evaluator(const std::string& model_file, bool use_gpu)
    : net(new PolicyValueNet(model_file, use_gpu)), stop(false){
    createHandlerThreads();
}
Evaluator::~Evaluator() { stop = true; qcv.notify_all(); handler.join(); }

bool Evaluator::evaluate(NNResultBuf* buf, const Game* game, HashValue hash) { // called by multiple threads. Cache must be thread-safe.
    // cache lookup
    bool cacheHit = cache.get(hash, buf->result);
    if(cacheHit) {
        //std::cerr << "cache hit\n";
        return true;
    }

    // enqueue request
    {
        std::lock_guard<std::mutex> lk(qmutex);
        q.push({buf, game, hash});
    }
    qcv.notify_one();

    // wait for result
    std::unique_lock<std::mutex> lk2(buf->resultmutex);
    buf->resultcv.wait(lk2, [&]{ return buf->result != nullptr; });
    return false;
}

bool Evaluator::asyncEvaluate(NNResultBuf* buf, const Game* game, HashValue hash){
    // cache lookup
    bool cacheHit = cache.get(hash, buf->result);
    if(cacheHit) {
        //std::cerr << "cache hit\n";
        return true;
    }

    // enqueue request
    {
        std::lock_guard<std::mutex> lk(qmutex);
        q.push({buf, game, hash});
    }
    qcv.notify_one();
    return false;
}

void Evaluator::updateModel(PolicyValueNet* updatedNet) {
    torch::NoGradGuard no_grad;
	auto src_state = updatedNet->policy_value_net->named_parameters();
	auto dst_state = net->policy_value_net->named_parameters();

	for (auto& item : src_state) {
		dst_state[item.key()].copy_(item.value());
	}
    cache.clear();
}
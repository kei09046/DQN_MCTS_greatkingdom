#include "evaluator.h"

void Evaluator::createHandlerThreads() {
    handler = std::thread(&Evaluator::HandlerWork, this);
}

void Evaluator::HandlerWork() {
    std::vector<const Game*> batchGames;
    std::vector<std::shared_ptr<NNResultBuf>> batchBufs;
    std::vector<HashValue> hashBufs;
    std::vector<std::unique_ptr<NNInput>> batchInputs;
    NNInput batchInput; // the requests' inputs back to back. Keeps its capacity between batches.

    while (!stop) {
        // wait until at least 1 request exists
        std::unique_lock<std::mutex> lk(qmutex);
        qcv.wait(lk, [&]{ return stop || !q.empty(); });

        // drain queue into batch
        while (!q.empty()) {
            evalRequest r = std::move(q.front()); q.pop();
            batchBufs.push_back(std::move(r.buf));
            hashBufs.push_back(r.hash);
            batchGames.push_back(r.game);
            batchInputs.push_back(std::move(r.input));
        }
        lk.unlock();

        // run batch : the inputs were built by the requesting threads, only merged here.
        batchInput.clear();
        for(const auto& in : batchInputs)
            batchInput.insert(batchInput.end(), in->begin(), in->end());
        std::vector<PolicyValueOutput> rawOutputs = net->batchEvaluate(batchInput, static_cast<int>(batchInputs.size()));
        std::vector<NodeInfo> outputs;
        outputs.reserve(rawOutputs.size());
        
        for(int i=0; i<rawOutputs.size(); ++i)
            outputs.push_back(convertForm(rawOutputs[i], *batchGames[i]));

        // try{
        //     outputs = net->batchEvaluate(batchGames);
        //     //std::cerr << "Batch evaluated successfully" << std::endl;
        // }catch(const c10::Error& e){
        //     std::cerr << "batch evaluate failed. retrying..." << std::endl << e.what() << std::endl;
        //     try{
        //         outputs = net->batchEvaluate(batchGames);
        //     }catch(const c10::Error& e2){
        //         std::cerr << "retry failed as well. attempting backupEvaluation" << std::endl;
        //         outputs = net->backupEvaluate(batchGames);
        //     }
        // }
        // std::cerr << "Batch evaluated successfully" << std::endl;

        // write results back & notify
        for (int i = 0; i < batchBufs.size(); i++) {
            auto buf = batchBufs[i];
            {
                std::lock_guard<std::mutex> lk2(buf->resultmutex);
                buf->result = std::make_shared<NodeInfo>(outputs[i]);
                cache.insert(hashBufs[i], buf->result);
                //buf->result = std::make_shared<NodeInfo>(outputs[i]);
            }
            buf->resultcv.notify_one();
        }

        hashBufs.clear();
        batchGames.clear();
        batchBufs.clear();
        batchInputs.clear();
    }
}

NodeInfo Evaluator::convertForm(const PolicyValueOutput& nnOutput, const Game& game){
    const auto& [logAct, winP, scoreEXP, scoreMap, captureMap] = nnOutput;

    // compute policy.(apply softmax + remove unplayable moves)
    const auto& availableMoves = game.getAvailableMoves();
    const auto& transferList = game.getTransferTable();
    const int moveSize = availableMoves.size();
    // available moves : [1, 2, 3, 7, 8, 9 ...]
    // transferList : [0, 0, 0, -1, -1, -1, 1, 0, 2, ...]

    std::vector<float> transferred(moveSize, std::numeric_limits<float>::lowest());
    for(int i=0; i<transferList.size(); ++i){
        if(transferList[i] != -1){
            transferred.at(transferList.at(i)) = std::max(transferred[transferList[i]], logAct.at(i));
        }
    }

    // std::vector<float> n_logit(moveSize);
    // for(int i=0; i<moveSize; ++i){
    //     n_logit[i] = transferred.at(availableMoves[i]);
    // }

    std::vector<float> expLogit(moveSize);
    float max_logit = *std::max_element(transferred.begin(), transferred.end()); // For numerical stability

    // Compute exponentials after subtracting max_logit
    float sum_exp = 0.0f;
    for (int i = 0; i < moveSize; ++i) {
        expLogit[i] = std::exp(transferred[i] - max_logit);
        sum_exp += expLogit[i];
    }

    // Normalize
    for (float& val : expLogit) {
        val /= sum_exp;
    }

    // compute other values
    const Color turn = game.getTurn();
    const Color turnScore = (turn == BLACK) ? BSCORE : WSCORE;
    const Color oppturnScore = (turn == BLACK) ? WSCORE : BSCORE;

    // float scoreV = 2 * ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi);

    // scoreV += scoreEXP;
    // for(int i=0; i<boardSize; ++i){
    //     if(game.getBoard({i / colSize, i % colSize}) == EMPTY){
    //         Color owned = game.getScoreBoard({i / colSize, i % colSize});
    //         scoreV += (owned == oppturnScore) ? 1.0f : ((owned == turnScore) ? -1.0f : scoreMap[i]);
    //     }
    // }

    float scoreV = ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi) + scoreEXP;

    // std::bitset<boardSize> mark;
    // for(int i=0; i<inputSize; ++i){
    //     int cidx = game.getChainIdx(i);
    //     const Chain c = game.getChain(i);

    //     if(c.size != 0 && !mark[cidx]){
    // 	    mark[cidx] = true;
    //         auto head = game.getStone({i / colSize, i % colSize}).head;
    //         auto cur = head;
    //         float avgCapChance = 0.0f;

    //         do {
    //             avgCapChance += captureMap[cur];
    //             cur = game.getStone({cur/colSize, cur%colSize}).next;
    //         } while (cur != head);

    //         avgCapChance /= c.size;

    //         int type = game.getBoard({i / colSize, i % colSize}) == turn ? 0 : 1;
    //         captureV[type] = (captureV[type] > avgCapChance) ? captureV[type] : avgCapChance;
    //     }
    // }

    // float utility = winP * 0.9f;
    // float utility = winP * 0.9f + (captureV[0] - captureV[1]) * 0.03f + scoreV * 0.01f;
    float utility = winP * 0.9f + scoreV * 0.01f;
    // float utility = winP;
    // if(globalConfig.detailedStat){
    //     static int cntr = 0;
    //     if(cntr++ % 100 == 0){
    //         game.displayBoardGUI(true);
    //         std::cout << "Q : " << winP << "\n";
    //         std::cout << "C : " << captureV[0] - captureV[1] << "\n";
    //         std::cout << "S : " << scoreV << "\n";
    //         std::cout << "U : " << utility << "\n";

    //         for(int i=0; i<rowSize; ++i){
    //             for(int j=0; j<colSize; ++j){
    //                 std::printf("%8.4f ", scoreMap[i * colSize + j]);
    //             }
    //             std::cout << std::endl;
    //         }
    //         std::cout << std::endl;
    //         for(int i=0; i<rowSize; ++i){
    //             for(int j=0; j<colSize; ++j){
    //                 std::printf("%8.4f ", captureMap[i * colSize + j]);
    //             }
    //             std::cout << std::endl;
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    return {expLogit, winP, utility, scoreEXP};
}

Evaluator::Evaluator(PolicyValueNet* net) : net(net), stop(false){
    createHandlerThreads();
}
Evaluator::Evaluator(const std::string& model_file, const std::string& model_type, int num_blocks, bool use_gpu)
    : net(new PolicyValueNet(model_file, model_type, num_blocks, use_gpu)), stop(false){
    createHandlerThreads();
}
Evaluator::Evaluator(const std::string& model_file, bool use_gpu)
    : net(new PolicyValueNet(model_file, use_gpu)), stop(false){
    createHandlerThreads();
}
Evaluator::~Evaluator() { stop = true; qcv.notify_all(); handler.join(); }

bool Evaluator::request(const std::shared_ptr<NNResultBuf>& buf, const Game* game, HashValue hash){ // called by multiple threads. Cache must be thread-safe.
    if(cache.get(hash, buf->result))
        return true;

    // the input is built here, in the requesting thread, so the handler thread only merges inputs.
    auto input = std::make_unique<NNInput>(PolicyValueNet::getData(*game));
    {
        std::lock_guard<std::mutex> lk(qmutex);
        q.push({buf, game, hash, std::move(input)});
    }
    qcv.notify_one();
    return false;
}

bool Evaluator::evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash) {
    if(request(buf, game, hash))
        return true;

    // wait for result
    std::unique_lock<std::mutex> lk2(buf->resultmutex);
    buf->resultcv.wait(lk2, [&]{ return buf->result != nullptr; });
    return false;
}

bool Evaluator::asyncEvaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash){
    return request(buf, game, hash);
}

void Evaluator::updateModel(PolicyValueNet* updatedNet) {
    torch::NoGradGuard no_grad;
	auto src_state = updatedNet->policy_value_net->named_parameters();
	auto dst_state = net->policy_value_net->named_parameters();

	for (auto& item : src_state) {
		dst_state[item.key()].copy_(item.value());
	}

	// BatchNorm params.
	auto src_buf = updatedNet->policy_value_net->named_buffers();
	auto dst_buf = net->policy_value_net->named_buffers();

	for (auto& item : src_buf) {
		dst_buf[item.key()].copy_(item.value());
	}
    net->fuseBatchNorm(); // the folded weights were computed from the old weights
    cache.clear();
}
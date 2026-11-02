#include "evaluator.h"

void Evaluator::create_handler_threads() {
    handler = std::thread(&Evaluator::handler_work, this);
}

void Evaluator::handler_work() {
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
            EvalRequest r = std::move(q.front()); q.pop();
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
        std::vector<PolicyValueOutput> rawOutputs = net->batch_evaluate(batchInput, static_cast<int>(batchInputs.size()), &pinned);
        std::vector<NodeInfo> outputs;
        outputs.reserve(rawOutputs.size());
        
        for(int i=0; i<rawOutputs.size(); ++i)
            outputs.push_back(convert_form(rawOutputs[i], *batchGames[i]));

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
    free_pinned_memory();
}

void Evaluator::free_pinned_memory(){
    pinned.free();
}

NodeInfo Evaluator::convert_form(const PolicyValueOutput& nnOutput, const Game& game){
    const auto& [logAct, winP, scoreEXP, scoreMap, captureMap] = nnOutput;

    // compute policy.(apply softmax + remove unplayable moves)
    const auto& availableMoves = game.get_available_moves();
    const auto& transferList = game.get_transfer_table();
    const int moveSize = availableMoves.size();
    // available moves : [1, 6, 8, ...]
    // transferList : [0, 0, 0, -1, -1, -1, 1, 0, 2, ...] (boardSize + 1)
    // all cords with same transferList value >= 0 shares one available move that makes it all territory.
    // transferList = -1 => illegal move(occupied etc.)
    // nth available move corresponds to transferList value n.

    std::vector<float> transferred(moveSize, std::numeric_limits<float>::lowest());
    for(int i=0; i<transferList.size(); ++i){
        if(transferList[i] != BASEGROUPID){
            transferred.at(transferList[i]) = std::max(transferred[transferList[i]], logAct.at(i));
        }
    }

    // std::vector<float> n_logit(moveSize);
    // for(int i=0; i<moveSize; ++i){
    //     n_logit[i] = transferred.at(availableMoves[i]);
    // }

    std::vector<float> expLogit(moveSize);
    float maxLogit = *std::max_element(transferred.begin(), transferred.end()); // For numerical stability

    // Compute exponentials after subtracting max_logit
    float sumExp = 0.0f;
    for (int i = 0; i < moveSize; ++i) {
        expLogit[i] = std::exp(transferred[i] - maxLogit);
        sumExp += expLogit[i];
    }

    // Normalize
    for (float& val : expLogit) {
        val /= sumExp;
    }

    // compute other values
    const Color turn = game.get_turn();
    const Color turnScore = (turn == BLACK) ? BSCORE : WSCORE;
    const Color oppturnScore = (turn == BLACK) ? WSCORE : BSCORE;

    float scoreV = 2 * ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi);

    scoreV += scoreEXP;
    for(int i=0; i<boardSize; ++i){
        if(game.get_board({i / colSize, i % colSize}) == EMPTY){
            Color owned = game.get_score_board({i / colSize, i % colSize});
            scoreV += (owned == oppturnScore) ? 1.0f : ((owned == turnScore) ? -1.0f : scoreMap[i]);
        }
    }

    //float scoreV = ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi) + scoreEXP;

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
    create_handler_threads();
}
Evaluator::Evaluator(const std::string& modelFile, const std::string& modelType, int numBlocks, bool useGpu)
    : net(new PolicyValueNet(modelFile, modelType, numBlocks, useGpu)), stop(false){
    create_handler_threads();
}
Evaluator::Evaluator(const std::string& modelFile, bool useGpu)
    : net(new PolicyValueNet(modelFile, useGpu)), stop(false){
    create_handler_threads();
}
Evaluator::~Evaluator() {
    {
        std::lock_guard<std::mutex> lk(qmutex); 
        stop = true;
    } 
    qcv.notify_all(); 
    handler.join(); 
}

bool Evaluator::request(const std::shared_ptr<NNResultBuf>& buf, const Game* game, HashValue hash){ // called by multiple threads. Cache must be thread-safe.
    if(cache.get(hash, buf->result))
        return true;

    // self play thread is responsible of building data.
    auto input = std::make_unique<NNInput>(PolicyValueNet::get_data(*game));
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

bool Evaluator::async_evaluate(std::shared_ptr<NNResultBuf> buf, const Game* game, HashValue hash){
    return request(buf, game, hash);
}

void Evaluator::update_model(PolicyValueNet* updatedNet) {
    torch::NoGradGuard noGrad;
	auto srcState = updatedNet->policyValueNet->named_parameters();
	auto dstState = net->policyValueNet->named_parameters();

	for (auto& item : srcState) {
		dstState[item.key()].copy_(item.value());
	}

	// BatchNorm params.
	auto srcBuf = updatedNet->policyValueNet->named_buffers();
	auto dstBuf = net->policyValueNet->named_buffers();

	for (auto& item : srcBuf) {
		dstBuf[item.key()].copy_(item.value());
	}
    net->fuse_batch_norm(); // the folded weights were computed from the old weights
    cache.clear();
}
#include "PMCTS.h"
#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "hash.h"
#include "evaluator.h"
#include "dirichlet.h"
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <chrono>
#include <ranges>
#include <execution>

#include "modelcompare.h"


namespace{
    const Hash hash;

    std::vector<float> softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves){
        assert(logit.size() == outputSize);

        std::vector<float> n_logit;
        n_logit.reserve(available_moves.size());
        for(const auto& move : available_moves){
            n_logit.push_back(logit[move.first * colSize + move.second]);
        }

        std::vector<float> exp_logit(n_logit.size());
        float max_logit = *std::max_element(n_logit.begin(), n_logit.end()); // For numerical stability

        // Compute exponentials after subtracting max_logit
        float sum_exp = 0.0f;
        for (int i = 0; i < n_logit.size(); ++i) {
            exp_logit[i] = std::exp(n_logit[i] - max_logit);
            sum_exp += exp_logit[i];
        }

        // Normalize
        for (float& val : exp_logit) {
            val /= sum_exp;
        }
        return exp_logit;
    }

    std::vector<float> softmax(const std::vector<float>& logit){
        std::vector<float> ret(logit.size());

        float maxLogit = *std::max_element(logit.begin(), logit.end());
        float sum = 0.0f;
        for (int i = 0; i < logit.size(); ++i) {
            ret[i] = std::exp(logit[i] - maxLogit);
            sum += ret[i];
        }
        for (int i = 0; i < 4; ++i) ret[i] /= sum; // apply softmax to get actual probability

        return ret;
    }

    std::pair<float, float> calculateQ(const std::shared_ptr<PolicyValueOutput> nnOutput, const Game& game)
    {
        const auto& [logAct, winP, scoreEXP, scoreMap, captureMap] = *nnOutput;

        float captureV[2] = {0.0f, 0.0f};
        float scoreV = std::reduce(std::execution::unseq, scoreMap.begin(), scoreMap.end()) + scoreEXP;

        const Color turn = game.getTurn();
        std::bitset<boardSize> mark;
        for(int i=0; i<inputSize; ++i){
            int cidx = game.getChainIdx(i);
		    const Chain c = game.getChain(i);

		    if(c.size != 0 && !mark[cidx]){
			    mark[cidx] = true;
                auto head = game.getStone({i / colSize, i % colSize}).head;
                auto cur = head;
                float avgCapChance = 0.0f;

                do {
                    avgCapChance += captureMap[i];
                    cur = game.getStone({cur/colSize, cur%colSize}).next;
                } while (cur != head);

                avgCapChance /= c.size;

                int type = game.getBoard({i / colSize, i % colSize}) == turn ? 0 : 1;
                captureV[type] = (captureV[type] > avgCapChance) ? captureV[type] : avgCapChance;
            }
        }

        float utility = winP * 0.75 + (captureV[0] - captureV[1]) * 0.25 + scoreV / boardSize;
        return {utility, winP};
    }
}

// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), S(0.0f), Wp(0.0f), forcedState(0), winmove(RESIGNMOVE), hashValue(hashValue), trans_table(trans_table){
}

void Node::addChild(const int r, const int c, const Game& ng){
    HashValue newHash = hash.computeHashAfterMove(game, {r, c}, hashValue);
    Node* childNode;

    if(globalConfig.transTable){
        if(trans_table->count(newHash) == 0){
            childNode = new Node(ng, newHash, trans_table);
            (*trans_table)[newHash] = childNode;
        }
        else{
            childNode = (*trans_table)[newHash];
        }
    }
    else{
        childNode = new Node(ng, newHash, trans_table);
    }

    child.push_back(childNode);
    available_moves.push_back({r, c});
}

void Node::expand(){
    std::bitset<outputSize> candidateLegal; // mark candidate legal moves

    // improve capture check performance by checking if there is any group with liberty count 1.
    Move threat = RESIGNMOVE;

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            const Chain c = game.getChain({i, j});
            if(c.size != 0 && c.liberties.count() == 1){
                auto color = game.getBoard({i, j});
                int onlyLib = c.liberties._Find_first();

                if(game.isLegal(onlyLib / colSize, onlyLib % colSize)){
                    // if my stone is under threat -> have to find only move unless can capture opponent's stone.
                    if(color == game.getTurn()){
                        threat = {onlyLib / colSize, onlyLib % colSize};
                    }

                    // if opponent stone is capturable
                    else{
                        winmove = {onlyLib / colSize, onlyLib % colSize};
                        forcedState = 2;
                        return;
                    }
                }
            }
        }
    }

    candidateLegal = game.getLegalMoves();
    // can only pass if it's beneficial
    candidateLegal[outputSize - 1] = (game.scoreWinner() == game.getTurn());

    std::vector<Game> nextGames(boardSize + 1); // +1 for pass
    // update scores & remove useless moves
    for(int idx = 0; idx < boardSize + 1; ++idx){
        if(candidateLegal[idx]){
            uint8_t r = idx / colSize;
            uint8_t c = idx % colSize;
            nextGames[idx] = game;
            auto [clr, wintype] = nextGames[idx].makeMove({r, c});

            if(clr == turn){ // there is immediate win by score. win in 1.
                forcedState = 2;
                winmove = {r, c};
                return;
            }

            // there is immediate capture next move, or the move is self-suicidal.
            else if((threat != RESIGNMOVE && (nextGames[idx].isLegal(threat) || wintype == CAPTURE))){
                candidateLegal[idx] = false;
            }

            else{
                candidateLegal &= nextGames[idx].getLegalMoves();
                candidateLegal[idx] = true; // keep itself
            }
        }
    }

    if(candidateLegal.none()){ // if there are no moves, mark it as loss.
        forcedState = -1;
        return;
    }
    
    // finally add child
    for(int idx = 0; idx < boardSize + 1; ++idx){
        if(candidateLegal[idx]){
            uint8_t r = idx / colSize;
            uint8_t c = idx % colSize;
            addChild(r, c, nextGames[idx]);
        }
    }

    assert(threat == RESIGNMOVE || candidateLegal.count() == 1);

    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

int Node::selectChildInSearch(){
    int maxi = -1;
    float pref, maxval = -5.0f; // pref may be less than -1.(due to score head)

    for(int i=0; i<available_moves.size(); ++i){
        int forced = child[i]->forcedState;
        
        // if winning continuation found.
        if(forced < 0){
            forcedState = -forced + (forced > 0 ? -1 : 1);
            winmove = available_moves[i];
            return i;
        }
        // only select non-losing move.
        else if(forced == 0){
            pref = ((edgeN[i] == 0.0f) ? ((globalConfig.fpu < 0.0f) ? 0.0f : -initQ-globalConfig.fpu) : child[i]->W / child[i]->N) 
            + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
            
            if(maxval < pref){
                maxval = pref; 
                maxi = i;
            }
        }
    }
    
    // if every move is lost, first move would be returned by default.
    if(maxi == -1){
        forcedState = -child[0]->forcedState - 1;
        return 0;
    }
    return maxi;
}

Move Node::selectMove(float temp){
    //std::cout << "available move size : " << available_moves.size() << std::endl;

    if(forcedState > 0){
        if(globalConfig.detailedStat)
            std::cout << "status: " << static_cast<int>(winmove.first) << " " << static_cast<int>(winmove.second)
            << " forced : " << -forcedState + (forcedState > 0 ? 1 : -1) << std::endl;
        return winmove;
    }
    else if(available_moves.size() == 0){
        return RESIGNMOVE;
    }
    else if(globalConfig.detailedStat){
        std::vector<int> v(available_moves.size());
        std::iota(v.begin(), v.end(), 0);
        std::sort(v.begin(), v.end(), [&](const int& a, const int& b){
            return child[a]->N > child[b]->N;
        });

        for(int i=0; i<std::min(static_cast<int>(available_moves.size()), 3); ++i){
            int idx = v[i];
            std::cout << "status: " << static_cast<int>(available_moves[idx].first) << " " << static_cast<int>(available_moves[idx].second)
            << " forced : " << child[idx]->forcedState << " sc: " << edgeN[idx] << " Q: " 
            << child[idx]->W/child[idx]->N << " initQ : " << child[idx]->initQ << " Wp : " << child[idx]->Wp/child[idx]->N 
            << " S : " << child[idx]->S / child[idx]->N << " P " << edgeP[idx] << std::endl;
        }
    }

    std::vector<float> weights(available_moves.size());
    std::vector<float> cumulative(available_moves.size());

    int maxi, maxn = -1, index;
    for(int i=0; i<available_moves.size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
        weights[i] = std::pow(edgeN[i], temp);
    }

    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

    if(temp < 5.0f){
        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        index = std::distance(cumulative.begin(), it);
        return available_moves[index];
    }

    return available_moves[maxi];
}


MoveData Node::selectMoveProb(float temp){
    std::vector<float> visitPortion(outputSize, 0.0f);

    if(forcedState > 0){
        visitPortion[winmove.first * colSize + winmove.second] = 1.0f;
        return {winmove, visitPortion};
    }
    if(available_moves.size() == 0){
        return {RESIGNMOVE, visitPortion};
    }
    std::vector<float> cumulative(available_moves.size()), weights(available_moves.size());
    int maxi, maxn = -1;
    for(int i=0; i<available_moves.size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
        weights[i] = std::pow(edgeN[i], temp);
        visitPortion[available_moves[i].first * colSize + available_moves[i].second] = edgeN[i]/N;
    }

    // std::cout << "visit portion" << std::endl;
    // for(int i=0; i<outputSize; ++i)
    //     std::cout << visitPortion[i] << " ";
    // std::cout << std::endl;

    if(temp < 5.0f){
        std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        int index = std::distance(cumulative.begin(), it);
        return {available_moves[index], visitPortion};
    }

    return {available_moves[maxi], visitPortion};
}

Node* Node::jump(Move move){
    if(N == 0){
        expand();
        N++;
    }

    int idx = -1;
    for(int i=0; i<available_moves.size(); ++i){
        if(available_moves[i] == move){
            idx = i;
            return child[idx];
        }
    }

    // if no child matches the move, add one. Only happens when human opponent makes suboptimal move.
    // std::cerr << "unexpected move!" << std::endl;
    // Game nGame = game;
    // nGame.makeMove(move);
    // addChild(move.first, move.second, nGame);
    // return child[child.size() - 1];

    std::cerr << "warning! jump to illegal location!" << std::endl;
    std::cerr << "requested move : " << static_cast<int>(move.first) << "," << static_cast<int>(move.second) << std::endl;
    std::cerr << "available options : " << std::endl;
    for(auto p : available_moves)
        std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";

    std::cerr << "node's state : " << std::endl;
    ModelCompare::displayBoardGUI(false, game);
    std::cerr << std::endl;

    return nullptr;
}

void Node::deleteTree(){
    for(Node* c : child){
        c->deleteTree();
    }
    delete this;
}

void Node::deleteTree(Node* exception){
    for(Node* c : child){
        if(c != exception)
            c->deleteTree();
    }
    delete this;
}

void Node::addDirichletNoise(Evaluator* evaluator){
    if (N == 0) {
        expand();
        if(forcedState == 0){
            auto buf = std::make_shared<NNResultBuf>();
            evaluator->evaluate(buf, &game, hashValue);

            edgeP = softmax(std::get<0>(*(buf->result)), available_moves);
            edgeN.assign(edgeP.size(), 0.0f);
        }
    }

    if(winmove == RESIGNMOVE && available_moves.size() > 0){
        std::vector<float> eta = sample_dirichlet(edgeP.size(), globalConfig.alpha); 
        for(int i=0; i<edgeP.size(); ++i)
            edgeP[i] = (1-globalConfig.eps) * edgeP[i] + globalConfig.eps * eta[i];
    }
}

MCTS::MCTS(Evaluator* evaluator) : 
evaluator(evaluator), trans_table(new std::unordered_map<HashValue, Node*>()){
    root = new Node(Game(), hash.baseHash(), trans_table);
    if(globalConfig.transTable){
        (*trans_table)[hash.baseHash()] = root;
    }
}

MCTS::MCTS(MCTS&& other) noexcept
    : root(other.root), evaluator(other.evaluator), trans_table(other.trans_table)
{
    other.root = nullptr;
    other.evaluator = nullptr;
    other.trans_table = nullptr;
}

MCTS::~MCTS(){
    delete trans_table;
}

void MCTS::runSimulation(const int playMode, const int nPlayout, const int timeLimit){
    //std::cout << "run simulation " << nPlayout << std::endl;
    if(globalConfig.dirichletNoise)
        root->addDirichletNoise(evaluator);

    int search_counter = 0;
    int evaluate_counter = 0;
    std::vector<Node*> current_evaluating_nodes;
    std::vector<std::vector<Node*>> need_update_chain;
    std::vector<std::shared_ptr<NNResultBuf>> result_buffer;
    bool stuck_during_search = false; // happens if meet evaluating node while searching

    if(playMode == PLAYOUT){
        while(evaluate_counter < nPlayout && root->forcedState == 0){
            playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
            playMode, nPlayout, timeLimit);
        }
        if(globalConfig.detailedStat)
            std::cout << "playout : " << search_counter << " " << evaluate_counter << std::endl;
    }
    else{
        auto duration = std::chrono::seconds(timeLimit);
        auto start = std::chrono::steady_clock::now();
        while(std::chrono::steady_clock::now() - start < duration && root->forcedState == 0){
            playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
            playMode, nPlayout, timeLimit);
        }
        if(globalConfig.detailedStat)
            std::cout << "playout : " << search_counter << " " << evaluate_counter << std::endl;
    }

    if(globalConfig.detailedStat)
        printVariation();
}

Move MCTS::getMove(float temp){
    runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout, globalConfig.time);
    return root->selectMove(temp);
}

MoveData MCTS::getMoveProb(float temp){
    runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout, globalConfig.time);
    return root->selectMoveProb(temp);
}

float MCTS::getEval(){
    if(root->N = 0)
        return 0.0f;
    return static_cast<float>(root->W) / root->N;
}

void MCTS::printVariation(){
    Node* node = root;

    while(node->N > 1){
        int maxv = -1;
        int maxi = -1;
        Move m;

        if(node->forcedState == 0){
            for(int i=0; i<node->available_moves.size(); ++i){
                // if(!globalConfig.transTable)
                //     assert(node->edgeN[i] == node->child[i]->N);
                // else
                //     assert(node->edgeN[i] <= node->child[i]->N);
                    
                if(node->edgeN[i] > maxv){
                    maxv = node->edgeN[i];
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;

            m = node->available_moves[maxi];
        }
        else if(node->forcedState > 0){
            if(node->forcedState == 2)
                break;

            m = node->winmove;
            assert(m != RESIGNMOVE);
            // std::cerr << "winmove : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;
            for(int i=0; i<node->available_moves.size(); ++i){
                if(node->available_moves[i] == m){
                    maxi = i;
                    break;
                }
            }
            // if(maxi == -1){
            //     for(const auto& options : node->available_moves)
            //         std::cerr << "options : " << static_cast<int>(options.first) << " " << static_cast<int>(options.second) << std::endl;
            // }
        }
        else{ // losing. Delay losing as much as possible.
            for(int i=0; i<node->available_moves.size(); ++i){
                if(node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }
            m = node->available_moves[maxi];
        }

        int visit = node->edgeN[maxi];
        node = node->child[maxi];
        std::cout << (int)m.first << " " << (int)m.second << " " << visit << " forced " << node->forcedState << " Q: " << 
            node->W/node->N << " initQ : " << node->initQ << " Wp : " << node->Wp/node->N 
            << " S : " << node->S / node->N << std::endl;  
    }
}

bool MCTS::jump(Move move){
    Node* old_root = root;
    root = root->jump(move);
    if(!globalConfig.transTable)
        old_root->deleteTree(root);
    return root != nullptr;
}

void MCTS::reset(){
    if(globalConfig.transTable){ // if transposition table is used, then all node is deleted when reset.
        for (auto& [hash, node] : *trans_table) {
            delete node;
        }
        trans_table->clear();
    }
    else{ // otherwise, parent nodes are deleted after each move is made.
        root->deleteTree();
    }

    root = new Node(Game(), hash.baseHash(), trans_table);

    if(globalConfig.transTable){
        (*trans_table)[hash.baseHash()] = root;
    }
}

void MCTS::playout(int& searchCounter, int& evaluateCounter, 
    std::vector<Node*>& inEvaluation, std::vector<std::vector<Node*>>& updateQueue,
    std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int nPlayout, const int timeLimit) {

    // SELECTION
    if((playMode == TIMEOUT || searchCounter < nPlayout) && (inEvaluation.size() < globalConfig.search_thread_num) && !searchStuck){
        std::vector<int> childIdx;
        std::vector<Node*> path;
        Node* cur = root;
        float evalQ = 0.0f, evalS = 0.0f, evalW = 0.0f;
        int forced = 0; // check if state is forced win / loss

        while (true) {
            path.push_back(cur);

            if (cur->N == 0.0f && (cur != root || !globalConfig.dirichletNoise)) { // first time visit
                cur->expand();
                forced = cur->forcedState;
                // if (cur->winmove != RESIGNMOVE){ // won
                //     evalQ = -1.0f;
                // }
                // else if(cur->forcedState < 0){ // lost
                //     evalQ = 1.0f;
                // }
                break;
            }

            forced = cur->forcedState;
            if(forced != 0)
                break;

            // if (cur->winmove != RESIGNMOVE){ // won
            //     evalQ = -1.0f;
            //     if(globalConfig.detailedStat){
            //         evalW = 0.0f;
            //         evalS = -cur->game.scoreDiff(cur->turn);
            //     }
            //     break;
            // }
            // if(cur->forcedState < 0){ // lost
            //     evalQ = 1.0f;
            //     if(globalConfig.detailedStat){
            //         evalW = 1.0f;
            //         evalS = -cur->game.scoreDiff(cur->turn);
            //     }
            //     break;
            // }

            if(std::find(inEvaluation.begin(), inEvaluation.end(), cur) != inEvaluation.end()){ // if node is evaluating, return
                searchStuck = true;
                return;
            }

            int a = cur->selectChildInSearch(); // assume node is evaluated
            childIdx.push_back(a);
            cur = cur->child[a];
        }

        searchCounter++;
        // set node visit stats
        for (Node* node : path) {
            node->N += 1.0f;
            node->W -= 1.0f; // apply VL
        }
        for(int i=0; i<childIdx.size(); ++i){
            path[i]->edgeN[childIdx[i]] += 1.0f;
        }

        if(forced != 0){ // if forced win/loss is found in leaf node, propagate that result immediately.
            evaluateCounter++;
            //propagate(path, childIdx, forced);

            if(forced > 0)
                propagate(path, -1.0f);
            else
                propagate(path, 1.0f);
        }
        else{ // if final search node is non-determined node, ask for evaluation
            // enqueue evaluation
            auto buf = std::make_shared<NNResultBuf>();

            bool cacheHit = evaluator->asyncEvaluate(buf, &(cur->game), cur->hashValue); // check cacheHit, also request eval
            if(!cacheHit){
                resultBuffer.push_back(buf);
                inEvaluation.push_back(cur);
                updateQueue.push_back(path);
            }
            else{
                std::vector<float> evalP = std::get<0>(*(buf->result));
                cur->edgeP = softmax(evalP, cur->available_moves);
                cur->edgeN = std::vector<float>(cur->edgeP.size(), 0.0f);

                if(globalConfig.detailedStat){ // if detailedStat = true, then update S, Wp variable. Else, ignore those values.
                    std::tie(cur->initQ, evalW) = calculateQ(buf->result, cur->game);
                    evalS = std::get<2>(*(buf->result));
                    evalQ = cur->initQ;
                }
                else{
                    evalQ = calculateQ(buf->result, cur->game).first;
                }

                // if eval is available right now, do param update right away.
                evaluateCounter++;
                propagate(path, evalQ, evalW, evalS);
            }
        }
    }

    //EVALUATION & UPDATE
    if(inEvaluation.size() >= globalConfig.search_thread_num || (playMode == PLAYOUT && searchCounter == nPlayout && !inEvaluation.empty()) ||
    (root->forcedState != 0 && !inEvaluation.empty()) || searchStuck){
        // wait for the result
        std::shared_ptr<NNResultBuf> rb = resultBuffer.at(inEvaluation.size() - 1);
        std::unique_lock<std::mutex> lk2(rb->resultmutex);
        rb->resultcv.wait(lk2, [&]{ return rb->result != nullptr; }); // wait until all evaluation queued are finished.

        for(int i=0; i<inEvaluation.size(); ++i){
            std::shared_ptr<NNResultBuf> buf = resultBuffer.at(i);
            std::vector<Node*> path = updateQueue.at(i);
            Node* cur = inEvaluation.at(i); 

            std::vector<float> evalP = std::get<0>(*(buf->result));
            cur->edgeP = softmax(evalP, cur->available_moves);
            cur->edgeN = std::vector<float>(cur->edgeP.size(), 0.0f);

            float evalQ = 0.0f, evalW = 0.0f, evalS = 0.0f;
            if(globalConfig.detailedStat){ // if detailedStat = true, then update S, Wp variable. Else, ignore those values.
                std::tie(cur->initQ, evalW) = calculateQ(buf->result, cur->game);
                //(cur->turn == BLACK ? globalConfig.komi : -globalConfig.komi)
                evalS = std::get<2>(*(buf->result));
                evalQ = cur->initQ;
            }
            else{
                evalQ = calculateQ(buf->result, cur->game).first;
            }

            // BACKUP (revert VL + add value)
            propagate(path, evalQ, evalW, evalS);
        }

        evaluateCounter += inEvaluation.size();
        resultBuffer.clear();
        updateQueue.clear();
        inEvaluation.clear();
        searchStuck = false;
    }
}

// void MCTS::propagate(const std::vector<Node*>& path, const std::vector<int>& childIdx, int forced){
//     assert(forced != 0);

//     Node* n;
//     // std::cerr << path.size() << " " << childIdx.size() << std::endl;
//     // std::cerr << "found forced sequence : " << forced << " ";
//     // for(int i=0; i<childIdx.size(); ++i)
//     //     std::cerr << static_cast<int>(path[i]->available_moves[childIdx[i]].first) << 
//     //     static_cast<int>(path[i]->available_moves[childIdx[i]].second) << " ";
//     // std::cerr << std::endl;

//     for(int i=childIdx.size() - 1; i >= 0; --i){
//         // on Node n, made move nextMove.
//         n = path[i];

//         if(forced < 0){ // child node is forced loss.
//             forced = -forced + (forced > 0 ? -1 : 1); // loss in 1 -> win in 2.
//             n->forcedState = forced;
//             n->winmove = (n->available_moves)[childIdx[i]]; // check winning move as only move
//         }
//         else{ // child node is forced win.
//             n->losingMoveCount++;

//             // every option is lost.
//             if(n->losingMoveCount == n->available_moves.size()){
//                 forced = -forced + (forced > 0 ? -1 : 1);
//                 n->forcedState = forced;
//             }
//             else{
//                 break;
//             }
//         }
//     }

//     // for(const Node* n : path){
//     //     std::cerr << n << " " << n->forcedState << std::endl;
//     // }
// }

void MCTS::propagate(const std::vector<Node*>& path, float evalQ, float evalW, float evalS){
    if(globalConfig.detailedStat){ // if detailedStat = true, update S, Wp variable as well. Otherwise, ignore those.
        for (Node* n : path | std::views::reverse) {
            n->W += 1.0f;   // revert VL
            n->W += evalQ;
            n->Wp += evalW;
            n->S += evalS;
            evalQ = -evalQ;
            evalS = -evalS;
            evalW = 1.0f-evalW;
        }
    }

    else{
        for (Node* n : path | std::views::reverse) {
            n->W += 1.0f;   // revert VL
            n->W += evalQ;
            evalQ = -evalQ;
        }
    }
}
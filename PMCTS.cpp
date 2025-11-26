#include "PMCTS.h"

#ifdef measureTime
thread_local size_t expandTime = 0; // expandTime = makeMoveTime + copyTime + extraTime
thread_local size_t evaluateTime = 0;
thread_local size_t searchTime = 0;
thread_local size_t makeMoveTime = 0;
thread_local size_t copyTime = 0;
thread_local size_t extraTime = 0;
thread_local size_t evalCacheInsertTime = 0;
thread_local size_t evalCacheFindTime = 0;
thread_local size_t evalCacheHit = 0;
thread_local size_t evalCacheNorotHit = 0;

std::vector<int> MCTS::getTimeStats() const{
    std::vector<int> stats;
    stats.reserve(10);
    stats[0] = expandTime;
    stats[1] = evaluateTime;
    stats[2] = searchTime;
    stats[3] = copyTime;
    stats[4] = makeMoveTime;
    stats[5] = extraTime;
    stats[6] = evalCacheHit;
    stats[7] = evalCacheNorotHit;
    stats[8] = evalCacheInsertTime;
    stats[9] = evalCacheFindTime;
    return stats;
}

void MCTS::resetTimeStats(){
    expandTime = evaluateTime = searchTime = copyTime = makeMoveTime = extraTime = 
    evalCacheHit = evalCacheNorotHit = evalCacheInsertTime = evalCacheFindTime = 0;
}
#endif

const Hash hash;

std::vector<float> Node::softmax(const std::vector<float>& logit, const std::vector<std::pair<int, int>>& available_moves){
    std::vector<float> n_logit;
    n_logit.reserve(available_moves.size());
    for(const auto& move : available_moves){
        n_logit.push_back(logit[move.first * colSize + move.second]);
    }

    std::vector<float> exp_logit(n_logit.size());
    float max_logit = *std::max_element(n_logit.begin(), n_logit.end()); // For numerical stability

    // Compute exponentials after subtracting max_logit
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n_logit.size(); ++i) {
        exp_logit[i] = std::exp(n_logit[i] - max_logit);
        sum_exp += exp_logit[i];
    }

    // Normalize
    for (float& val : exp_logit) {
        val /= sum_exp;
    }
    return exp_logit;
}

// N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
Node::Node(const Game& g, const HashValue hashValue, EvalCache* const eval_cache, std::unordered_map<HashValue, Node*>* const trans_table):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), P(0.0f), initQ(0.0f), winmove({-1, -1}), hashValue(hashValue), eval_cache(eval_cache), trans_table(trans_table){
}

void Node::expand(){
    #ifdef measureTime
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    #endif

    color clr;

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            if(game.isLegal(i, j)){
                #ifdef measureTime
                std::chrono::steady_clock::time_point copyBegin = std::chrono::steady_clock::now();
                #endif
                Game ng = game;

                #ifdef measureTime
                std::chrono::steady_clock::time_point copyEnd = std::chrono::steady_clock::now();
                copyTime += (std::chrono::duration_cast<std::chrono::microseconds>(copyEnd - copyBegin).count());

                std::chrono::steady_clock::time_point moveBegin = std::chrono::steady_clock::now();
                #endif

                clr = ng.makeMove(i, j);

                #ifdef measureTime
                std::chrono::steady_clock::time_point moveEnd = std::chrono::steady_clock::now();
                makeMoveTime += (std::chrono::duration_cast<std::chrono::microseconds>(moveEnd - moveBegin).count());

                std::chrono::steady_clock::time_point extraBegin = std::chrono::steady_clock::now();
                #endif

                if(clr == EMPTY){
                    HashValue newHash = hash.computeHashAfterMove(game, {i, j}, hashValue);
                    Node* childNode;

                    if(trans_table->count(newHash) == 0){
                        childNode = new Node(ng, newHash, eval_cache, trans_table);
                        (*trans_table)[newHash] = childNode;
                    }
                    else{
                        childNode = (*trans_table)[newHash];
                    }
                    child.push_back(childNode);
                    available_moves.push_back({i, j});
                }

                else if(clr == turn){
                    winmove = {i, j};
                    #ifdef measureTime
                    std::chrono::steady_clock::time_point extraEnd = std::chrono::steady_clock::now();
                    extraTime += (std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - extraBegin).count());
                    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - begin).count());
                    #endif
                    return;
                }

                #ifdef measureTime
                std::chrono::steady_clock::time_point extraEnd = std::chrono::steady_clock::now();
                extraTime += (std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - extraBegin).count());
                #endif
            }
        }
    }

    if(game.scoreWinner() == game.getTurn()){ // can pass only if it's beneficial
        Game pass = game;
        pass.makeMove(rowSize, 0);

        HashValue newHash = hash.computeHashAfterMove(game, {rowSize, 0}, hashValue);
        Node* childNode;

        if(trans_table->count(newHash) == 0){
            childNode = new Node(pass, newHash, eval_cache, trans_table);
            (*trans_table)[newHash] = childNode;
        }
        else{
            childNode = (*trans_table)[newHash];
        }

        child.push_back(childNode);
        available_moves.push_back({rowSize, 0}); // pass
    }

    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

float Node::searchandPropagate(PolicyValueNet& net){
    if(N++ == 0){
        expand();
    }
    
    if(winmove.first >= 0){ // position is won
        W--;
        return 1.0f;
    }
    if(available_moves.size() == 0){ // position is lost
        W++;
        return -1.0f;
    }

    if(N == 1){
        #ifdef measureTime
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        std::shared_ptr<PolicyValueOutput> entry;

        #ifdef measureTime
            std::chrono::steady_clock::time_point find_begin = std::chrono::steady_clock::now();
        #endif
        bool cacheHit = eval_cache->get(hashValue, entry);
        #ifdef measureTime
            std::chrono::steady_clock::time_point find_end = std::chrono::steady_clock::now();
            evalCacheFindTime += (std::chrono::duration_cast<std::chrono::microseconds>(find_end - find_begin).count());
        #endif

        if(!cacheHit){ // cache miss
            // entry = net.evaluate(game);
            // instead, compute chlidren nodes at once.
            std::vector<const Game*> gameBatch;
            gameBatch.reserve((1 + available_moves.size()));

            gameBatch.push_back(&game);
            for(auto c : child)
                gameBatch.push_back(&(c->game));

            std::vector<PolicyValueOutput> entries = net.batchEvaluate(gameBatch);
            entry = std::make_shared<PolicyValueOutput>(entries[0]);

            #ifdef measureTime
            std::chrono::steady_clock::time_point insert_begin = std::chrono::steady_clock::now();
            #endif

            // std::cerr << "original policyvalue output : " << std::endl;
            // for(float x : entries[0].first)
            //     std::cerr << x << " ";
            // std::cerr << std::endl;
            
            eval_cache->insert(hashValue, entry);

            for(size_t i=0; i<available_moves.size(); ++i){
                auto ce = std::make_shared<PolicyValueOutput>(entries[i+1]);
                eval_cache->insert(child[i]->hashValue, ce);
            }
            
            #ifdef measureTime
            std::chrono::steady_clock::time_point insert_end = std::chrono::steady_clock::now();
            evalCacheInsertTime += std::chrono::duration_cast<std::chrono::microseconds>(insert_end - insert_begin).count();
            #endif
        }
        #ifdef measureTime
        else{
            evalCacheHit++;
        }
        #endif

        auto& logp = entry->first;
        auto q = entry->second;

        #ifdef measureTime
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        evaluateTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        #endif

        std::vector<float> p = softmax(logp, available_moves);

        for(int i=0; i<available_moves.size(); ++i){
            child[i]->P = p[i];
        }

        initQ = q;
        W += q;
        return -q;
    }
    

    int maxi = -1;
    float pref, maxval = -1.0f;

    #ifdef measureTime
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    #endif
    for(int i=0; i<available_moves.size(); ++i){
        pref = ((child[i]->N == 0) ? 0.0f : child[i]->W / child[i]->N) + cPuct * child[i]->P * sqrt(N)/(1 + child[i]->N);
        
        if(maxval < pref){
            maxval = pref; 
            maxi = i;
        }
    }
    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    searchTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif

    float r = child[maxi]->searchandPropagate(net); // segfault?
    W += r;
    return -r;
}

std::pair<int, int> Node::selectMove(float temp){
    if(winmove.first >= 0)
        return winmove;
    if(available_moves.size() == 0){ // if lost, resign
        return {-1, -1};
    }

    std::vector<float> weights(available_moves.size());
    std::vector<float> cumulative(available_moves.size());

    int maxi, maxn = -1, index;
    for(int i=0; i<available_moves.size(); ++i){
        if(child[i]->N > maxn){
            maxn = child[i]->N;
            maxi = i;
        }
        weights[i] = std::pow(child[i]->N, temp);
    }

    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

    if(temp < 5.0f){
        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        index = std::distance(cumulative.begin(), it);
        return available_moves[index];
    }

    for(int i=0; i<available_moves.size(); ++i){
        std::cout << "move : " << available_moves[i].first << " " << available_moves[i].second << " sc: " << child[i]->N << " wc: " << 
        child[i]->W << " initQ : " << child[i]->initQ << " P " << child[i]->P << std::endl;
    }
    return available_moves[maxi];
}

MoveData Node::selectMoveProb(float temp){
    std::array<float, outputSize> visitPortion;
    visitPortion.fill(0.0f);

    if(winmove.first >= 0)
        return {winmove, visitPortion};
    if(available_moves.size() == 0){ // if lost, resign
        return {{-1, -1}, visitPortion};
    }

    std::vector<float> cumulative(available_moves.size()), weights(available_moves.size());
    int maxi, maxn = -1;
    for(int i=0; i<available_moves.size(); ++i){
        if(child[i]->N > maxn){
            maxn = child[i]->N;
            maxi = i;
        }
        weights[i] = std::pow(child[i]->N, temp);
        visitPortion[available_moves[i].first * colSize + available_moves[i].second] = child[i]->N/N;
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
        size_t index = std::distance(cumulative.begin(), it);

        // std::cout << "make move : " << available_moves[index].first << " " << available_moves[index].second << " win count : " << child[index]->W << " visit count : " << child[index]->N <<
        // " prob : " << child[index]->P << " eval : " << child[index]->initQ << "\n";

        return {available_moves[index], visitPortion};
    }

    // std::cout << "make move : " << available_moves[maxi].first << " " << available_moves[maxi].second << " win count : " << child[maxi]->W << " visit count : " << child[maxi]->N << 
    // " prob : " << child[maxi]->P << " eval : " << child[maxi]->initQ << "\n";

    return {available_moves[maxi], visitPortion};
}

Node* Node::jump(std::pair<int, int> move){
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

    std::cerr << "warning! jump to illegal location!" << std::endl;
    std::cerr << "requested move : " << move.first << "," << move.second << std::endl;
    game.displayBoardGUI();
    std::cout << std::endl;
    std::cerr << "available options : " << std::endl;
    for(auto p : available_moves)
        std::cerr << p << " ";

    return nullptr;
}

void MCTS::runSimulation(){
    for(int i=0; i<playout; ++i){
        //std::cout << "on playout " << i << std::endl;
        root->searchandPropagate(*net);
    }
}

std::pair<int, int> MCTS::getMove(float temp){
    runSimulation();
    return root->selectMove(temp);
}

MoveData MCTS::getMoveProb(float temp){
    runSimulation();
    return root->selectMoveProb(temp);
}

bool MCTS::jump(std::pair<int, int> move){
    root = root->jump(move);
    return root != nullptr;
}

void MCTS::reset(){
    for (auto& [hash, node] : *trans_table) {
        delete node;
    }
    trans_table->clear();
    root = new Node(Game(), hash.baseHash(), eval_cache, trans_table);
    (*trans_table)[hash.baseHash()] = root;
}

void MCTS::updateModel(){
    eval_cache->clear();
}

MCTS::MCTS(int playout, PolicyValueNet* net, EvalCache* const eval_cache, std::unordered_map<HashValue, Node*>* const trans_table) : 
net(net), playout(playout), eval_cache(eval_cache), trans_table(trans_table){
    root = new Node(Game(), hash.baseHash(), eval_cache, trans_table);
    (*trans_table)[hash.baseHash()] = root;
}
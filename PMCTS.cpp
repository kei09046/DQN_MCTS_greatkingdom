#include "PMCTS.h"

#ifdef measureTime
thread_local size_t expandTime = 0; // expandTime = makeMoveTime + copyTime + extraTime
thread_local size_t evaluateTime = 0;
thread_local size_t searchTime = 0;
thread_local size_t makeMoveTime = 0;
thread_local size_t copyTime = 0;
thread_local size_t extraTime = 0;
thread_local size_t evalCacheHit = 0;
thread_local size_t evalCacheNorotHit = 0;

std::vector<int> MCTS::getTimeStats() const{
    std::vector<int> stats;
    stats.reserve(8);
    stats[0] = expandTime;
    stats[1] = evaluateTime;
    stats[2] = searchTime;
    stats[3] = copyTime;
    stats[4] = makeMoveTime;
    stats[5] = extraTime;
    stats[6] = evalCacheHit;
    stats[7] = evalCacheNorotHit;
    return stats;
}

void MCTS::resetTimeStats(){
    expandTime = evaluateTime = searchTime = copyTime = makeMoveTime = extraTime = evalCacheHit = evalCacheNorotHit = 0;
}
#endif

const Hash hash;
thread_local std::unordered_map<HashValue, Node*> trans_table;


std::vector<float> Node::softmax(std::vector<float>& logit){
    std::vector<float> exp_logit(logit.size());
    float max_logit = *std::max_element(logit.begin(), logit.end()); // For numerical stability

    // Compute exponentials after subtracting max_logit
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logit.size(); ++i) {
        exp_logit[i] = std::exp(logit[i] - max_logit);
        sum_exp += exp_logit[i];
    }

    // Normalize
    for (float& val : exp_logit) {
        val /= sum_exp;
    }

    return exp_logit;
}

// N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
Node::Node(const Game& g, const HashValue hashValue, EvalCache<PolicyValueOutput>* const eval_cache): game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), P(0.0f), initQ(0.0f), winmove({-1, -1}), hashValue(hashValue), eval_cache(eval_cache){
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

                    if(trans_table.count(newHash) == 0){
                        childNode = new Node(ng, newHash, eval_cache);
                        trans_table[newHash] = childNode;
                    }
                    else{
                        childNode = trans_table[newHash];
                    }
                    child.push_back(childNode);
                    legal.push_back({i, j});
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

        if(trans_table.count(newHash) == 0){
            childNode = new Node(pass, newHash, eval_cache);
            trans_table[newHash] = childNode;
        }
        else{
            childNode = trans_table[newHash];
        }

        child.push_back(childNode);
        legal.push_back({rowSize, 0}); // pass
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
    if(legal.size() == 0){ // position is lost
        W++;
        return -1.0f;
    }

    if(N == 1){
        #ifdef measureTime
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        //std::vector<PolicyValueOutput> entry;

        // auto m = std::min_element(hashValues.begin(), hashValues.end());
        // auto minHash = *m;
        // auto index = std::distance(hashValues.begin(), m);

        // if(!eval_cache.get(minHash, entry)){
        //     auto eval = net.evaluate(&game, legal);                       
        //     auto [base_eval, base_legal] = rotateNNOutputandLegal(eval, legal, index, rowSize); // first rotate into minHash state
        //     entry = rotateAllNNOutputs(base_eval, base_legal, rowSize);           // make all rotations based on it
        //     eval_cache.insert(minHash, entry);
        // }
        // else if(entry[0].first.size() != legal.size()){ // hash collision
        //     std::cout << "warning! hash collision! Hash value : " << minHash;
        //     entry = rotateAllNNOutputs(net.evaluate(&game, legal), legal, rowSize);
        // }
        // else{
        //     evalCacheHit++; // for debugging
        // }

        PolicyValueOutput entry;
        if(!eval_cache->get(hashValue, entry)){
            entry = net.evaluate(&game, legal);                       
            eval_cache->insert(hashValue, entry);
        }
        else if(entry.first.size() != legal.size()){ // hash collision
            std::cout << "warning! hash collision! Hash value : " << hashValue << std::endl;
            entry = net.evaluate(&game, legal);
        }
        #ifdef measureTime
        else{
            evalCacheNorotHit++;
        }
        #endif

        // auto logp = entry[(8-index)%8].first;
        // auto q = entry[(8-index)%8].second;
        auto logp = entry.first;
        auto q = entry.second;

        #ifdef measureTime
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        evaluateTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        #endif

        auto p = softmax(logp);

        for(int i=0; i<legal.size(); ++i){
            child[i]->P = p[i];
        }
        initQ = q;
        W += q;
        return -q;
    }
    

    int maxi;
    float pref, maxval = -1.0f;

    #ifdef measureTime
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    #endif
    for(int i=0; i<legal.size(); ++i){
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

    float r = child[maxi]->searchandPropagate(net);
    W += r;
    return -r;
}

std::pair<int, int> Node::selectMove(float temp){
    if(winmove.first >= 0)
        return winmove;
    if(legal.size() == 0){ // if lost, resign
        return {-1, -1};
    }

    std::vector<float> weights(legal.size());
    std::vector<float> cumulative(legal.size());

    int maxi, maxn = -1, index;
    for(int i=0; i<legal.size(); ++i){
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
        return legal[index];
    }

    for(int i=0; i<legal.size(); ++i){
        std::cout << "move : " << legal[i].first << " " << legal[i].second << " sc: " << child[i]->N << " wc: " << 
        child[i]->W << " initQ : " << child[i]->initQ << " P " << child[i]->P << std::endl;
    }
    return legal[maxi];
}

MoveData Node::selectMoveProb(float temp){
    std::array<float, outputSize> visitPortion;
    visitPortion.fill(0.0f);

    if(winmove.first >= 0)
        return {winmove, visitPortion};
    if(legal.size() == 0){ // if lost, resign
        return {{-1, -1}, visitPortion};
    }

    std::vector<float> cumulative(legal.size()), weights(legal.size());
    int maxi, maxn = -1;
    for(int i=0; i<legal.size(); ++i){
        if(child[i]->N > maxn){
            maxn = child[i]->N;
            maxi = i;
        }
        weights[i] = std::pow(child[i]->N, temp);
        visitPortion[legal[i].first * colSize + legal[i].second] = child[i]->N/N;
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

        // std::cout << "make move : " << legal[index].first << " " << legal[index].second << " win count : " << child[index]->W << " visit count : " << child[index]->N <<
        // " prob : " << child[index]->P << " eval : " << child[index]->initQ << "\n";

        return {legal[index], visitPortion};
    }

    // std::cout << "make move : " << legal[maxi].first << " " << legal[maxi].second << " win count : " << child[maxi]->W << " visit count : " << child[maxi]->N << 
    // " prob : " << child[maxi]->P << " eval : " << child[maxi]->initQ << "\n";

    return {legal[maxi], visitPortion};
}

Node* Node::jump(std::pair<int, int> move){
    if(N == 0){
        expand();
        N++;
    }

    int idx = -1;
    for(int i=0; i<legal.size(); ++i){
        if(legal[i] == move){
            idx = i;
            return child[idx];
        }
    }

    std::cerr << "warning! jump to illegal location!" << std::endl;
    std::cerr << "requested move : " << move.first << "," << move.second << std::endl;
    game.displayBoardGUI();
    std::cout << std::endl;
    std::cerr << "available options : " << std::endl;
    for(auto p : legal)
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
    for (auto& [hash, node] : trans_table) {
        delete node;
    }
    trans_table.clear();
    root = new Node(Game(), hash.baseHash(), eval_cache);
    trans_table[hash.baseHash()] = root;
}

void MCTS::updateModel(){
    eval_cache->clear();
}

MCTS::MCTS(int playout, PolicyValueNet* net, EvalCache<PolicyValueOutput>* const eval_cache) : net(net), playout(playout), eval_cache(eval_cache){
    root = new Node(Game(), hash.baseHash(), eval_cache);
    trans_table[hash.baseHash()] = root;
}
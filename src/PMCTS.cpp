#include "PMCTS.h"

const Hash hash;

std::vector<float> softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves){
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

std::pair<float, float> calculateQ(const std::vector<float>& winLogit, const std::vector<float>& scoreDist, float komi, float score_factor, float risk_aversion)
{
    // softmax
    float maxLogit = *std::max_element(winLogit.begin(), winLogit.end());
    float sum = 0.0f;
    float p[4];
    for (int i = 0; i < 4; ++i) {
        p[i] = std::exp(winLogit[i] - maxLogit);
        sum += p[i];
    }
    for (int i = 0; i < 4; ++i) p[i] /= sum; // apply softmax to get actual probability

    float p_win  = p[0] + p[1];
    float p_loss = p[2] + p[3];
    
    if(true/*p[0] + p[2] < 0.1f*/) // if probability of winning by score is less than 0.1, only use policy output.
        return {2 * p_win - 1.0f, p_win};

    // Step 1: compute mean
    float score_mean = 0.0f;
    for (int i = 0; i < 31; ++i)
    {
        int score = i - 15;   // map index 0..30 to score -15..15
        score_mean += score * scoreDist[i];
    }

    // Step 2: compute variance
    float score_var = 0.0f;
    for (int i = 0; i < 31; ++i)
    {
        int score = i - 15;
        float diff = score - score_mean;
        score_var += diff * diff * scoreDist[i];
    }
    float score_std = std::sqrt(score_var);

    // Step 1: convert score to utility relative to komi
    float score_util = score_mean - komi;

    // Step 2: scale score utility to match value scale
    score_util *= score_factor;

    // Step 3: optional risk aversion penalty
    float risk_penalty = risk_aversion * score_std;

    // Step 4: combine win probability and score utility
    // A simple linear combination
    float utility = p_win + (score_util - risk_penalty) * (p[0] + p[2]);

    // Step 5: clamp utility to [0,1] if desired
    utility = std::clamp(utility, 0.0f, 1.0f);

    return {2 * utility - 1.0f, p_win};
}



// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), initS(0.0f), initW(0.0f), winmove(RESIGNMOVE), hashValue(hashValue), trans_table(trans_table){
}

void Node::addChild(int r, int c, Game ng){
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
    Color clr;
    std::vector<Game> nextGames(boardSize + 1, game); // +1 for pass
    std::bitset<outputSize> candidateLegal; // mark candidate legal moves

    // improve performance by first checking capture for all moves -> calculate territory for all moves -> reduce options
    for(uint8_t i=0; i<rowSize; ++i){
        for(uint8_t j=0; j<colSize; ++j){
            if(game.isLegal(i, j)){
                clr = nextGames[i * colSize + j].makeMoveNoScoreUpdate({i, j}); // only check if capture occurs

                if(clr == EMPTY){
                    candidateLegal[i * colSize + j] = true;
                }

                else if(clr == turn){
                    winmove = {i, j};
                    return;
                }
            }
        }
    }

    if(game.scoreWinner() == game.getTurn()){ // can pass only if it's beneficial
        nextGames[boardSize].makeMoveNoScoreUpdate(PASSMOVE);
        candidateLegal[boardSize] = true;
    }

    if(game.getMoveCount() >= 2){ // if after second move, update scores & remove useless moves
        for(size_t idx = 0; idx < boardSize + 1; ++idx){
            if(candidateLegal[idx]){
                uint8_t r = idx / colSize;
                uint8_t c = idx % colSize;
                clr = nextGames[idx].updateScoreAfter({r, c});

                if(clr == turn){
                    winmove = {r, c};
                    return;
                }

                else if(clr == EMPTY){
                    candidateLegal &= nextGames[idx].getLegalMoves();
                    candidateLegal[idx] = true; // keep itself
                }
            }
        }
    }

    // finally add child
    for(size_t idx = 0; idx < boardSize + 1; ++idx){
        if(candidateLegal[idx]){
            uint8_t r = idx / colSize;
            uint8_t c = idx % colSize;
            addChild(r, c, nextGames[idx]);
        }
    }

    // for(Move m : available_moves){
    //     std::cerr << "available move after expand : " << static_cast<int>(m.first) << "," << static_cast<int>(m.second) << std::endl;
    // }
    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

int Node::selectChildInSearch(){
    int maxi = 0;
    float pref, maxval = -1.0f;

    for(int i=0; i<available_moves.size(); ++i){
        pref = ((edgeN[i] == 0.0f) ? ((globalConfig.fpu < 0.0f) ? 0.0f : -initQ-globalConfig.fpu) : child[i]->W / child[i]->N) 
        + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
        
        if(maxval < pref){
            maxval = pref; 
            maxi = i;
        }
    }
    return maxi;
}

Move Node::selectMove(float temp){
    if(winmove != RESIGNMOVE)
        return winmove;
    if(available_moves.size() == 0){ // if lost, resign
        return RESIGNMOVE;
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

    std::cout << "available move size : " << available_moves.size() << std::endl;
    for(int i=0; i<available_moves.size(); ++i){
        if(child[i]->N != 0.0f){
            std::cout << "status: " << static_cast<int>(available_moves[i].first) << " " << static_cast<int>(available_moves[i].second) << 
            " sc: " << edgeN[i] << " Q: " << 
            child[i]->W/child[i]->N << " initQ : " << child[i]->initQ << " W : " << child[i]->initW << " S : " << child[i]->initS << " P " << edgeP[i] << std::endl;
        }
    }
    return available_moves[maxi];
}

MoveData Node::selectMoveProb(float temp){
    std::vector<float> visitPortion(outputSize, 0.0f);

    if(winmove != RESIGNMOVE)
        return {winmove, visitPortion};
    if(available_moves.size() == 0){ // if lost, resign
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
        size_t index = std::distance(cumulative.begin(), it);
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

    std::cerr << "warning! jump to illegal location!" << std::endl;
    std::cerr << "requested move : " << move.first << "," << move.second << std::endl;
    std::cerr << "available options : " << std::endl;
    for(auto p : available_moves)
        std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";

    return nullptr;
}

#ifndef transTable
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
#endif

void Node::addDirichletNoise(Evaluator* evaluator){
    if (N == 0) {
        expand();
        if(winmove == RESIGNMOVE && available_moves.size() > 0){
            NNResultBuf buf;
            evaluator->evaluate(buf, &game, hashValue);

            edgeP = softmax(std::get<0>(*buf.result), available_moves);
            edgeN.assign(edgeP.size(), 0.0f);
        }
    }

    if(winmove == RESIGNMOVE && available_moves.size() > 0){
        std::vector<float> eta = sample_dirichlet(edgeP.size(), globalConfig.alpha); 
        for(int i=0; i<edgeP.size(); ++i)
            edgeP[i] = (1-globalConfig.eps) * edgeP[i] + globalConfig.eps * eta[i];
    }
}

MCTS::MCTS(int playout, Evaluator* evaluator) : 
nPlayout(playout), evaluator(evaluator), trans_table(new std::unordered_map<HashValue, Node*>()){
    root = new Node(Game(), hash.baseHash(), trans_table);
    if(globalConfig.transTable){
        (*trans_table)[hash.baseHash()] = root;
    }
}

MCTS::MCTS(MCTS&& other) noexcept
    : root(other.root), nPlayout(other.nPlayout),
      evaluator(other.evaluator), trans_table(other.trans_table)
{
    other.root = nullptr;
    other.evaluator = nullptr;
    other.trans_table = nullptr;
}

MCTS::~MCTS(){
    delete trans_table;
}

void MCTS::runSimulation(){
    //std::cout << "run simulation " << nPlayout << std::endl;
    if(globalConfig.dirichletNoise)
        root->addDirichletNoise(evaluator);

    int search_counter = 0;
    int evaluate_counter = 0;
    std::vector<Node*> current_evaluating_nodes;
    std::vector<std::vector<Node*>> need_update_chain;
    std::vector<NNResultBuf*> result_buffer;
    bool stuck_during_search = false; // happens if meet evaluating node while searching

    while(evaluate_counter < nPlayout){
        playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search);
    }
}

Move MCTS::getMove(float temp){
    runSimulation();
    return root->selectMove(temp);
}

MoveData MCTS::getMoveProb(float temp){
    runSimulation();
    return root->selectMoveProb(temp);
}

bool MCTS::jump(Move move){
    Node* old_root = root;
    root = root->jump(move);
    #ifndef transTable
    old_root->deleteTree(root);
    #endif
    return root != nullptr;
}

void MCTS::reset(){
    if(globalConfig.transTable){ // if transposition table is used, then all node is deleted when reset.
        for (auto& [hash, node] : *trans_table) {
            delete node;
        }
        trans_table->clear();
    }
    else{ // otherwise, nodes are deleted after each move is made.
        root->deleteTree();
    }

    root = new Node(Game(), hash.baseHash(), trans_table);

    if(globalConfig.transTable){
        (*trans_table)[hash.baseHash()] = root;
    }
}

void MCTS::playout(int& searchCounter, int& evaluateCounter, 
    std::vector<Node*>& inEvaluation, std::vector<std::vector<Node*>>& updateQueue,
    std::vector<NNResultBuf*>& resultBuffer, bool& searchStuck) {

    // SELECTION
    if((searchCounter < nPlayout) && (inEvaluation.size() < globalConfig.search_thread_num) && !searchStuck){
        std::vector<int> childIdx;
        std::vector<Node*> path;
        Node* cur = root;
        float evalQ = 0.0f;

        while (true) {
            path.push_back(cur);

            if (cur->N == 0.0f && (cur != root || !globalConfig.dirichletNoise)) { // first time visit
                cur->expand();
                if (cur->winmove != RESIGNMOVE){ // won
                    evalQ = -1.0f;
                }
                else if(cur->available_moves.size() == 0){ // lost
                    evalQ = 1.0f;
                }
                break;
            }

            if (cur->winmove != RESIGNMOVE){ // won
                evalQ = -1.0f;
                break;
            }
            if(cur->available_moves.size() == 0){ // lost
                evalQ = 1.0f;
                break;
            }

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

        if(evalQ == 0.0f){ // if final search node is non-terminal, not evaluated node
            // enqueue evaluation
            NNResultBuf* buf = new NNResultBuf();

            bool cacheHit = evaluator->asyncEvaluate(*buf, &(cur->game), cur->hashValue); // check cacheHit, also request eval
            if(!cacheHit){
                resultBuffer.push_back(buf);
                inEvaluation.push_back(cur);
                updateQueue.push_back(path);
            }
            else{
                std::tie(cur->initQ, cur->initW) = calculateQ(std::get<1>(*(buf->result)), std::get<3>(*(buf->result)));
                cur->initS = std::get<2>(*(buf->result));
                std::vector<float> evalP = std::get<0>(*(buf->result));
                cur->edgeP = softmax(evalP, cur->available_moves);
                cur->edgeN = std::vector<float>(cur->edgeP.size(), 0.0f);
                evalQ = cur->initQ;

                delete buf;
            }
        }

        if(evalQ != 0.0f){ // if eval is available right now, do param update right away.
            evaluateCounter++;
            path[path.size() - 1]->initQ = evalQ;
            for (int i = path.size() - 1; i >= 0; --i) {
                Node* n = path[i];
                n->W += 1.0f;   // revert VL
                n->W += evalQ;
                evalQ = -evalQ;
            }
        }
    }

    //EVALUATION & UPDATE
    if(inEvaluation.size() >= globalConfig.search_thread_num || (searchCounter == nPlayout && !inEvaluation.empty()) || searchStuck){
        // wait for the result
        NNResultBuf* rb = resultBuffer[inEvaluation.size() - 1];
        std::unique_lock<std::mutex> lk2(rb->resultmutex);
        rb->resultcv.wait(lk2, [&]{ return rb->result != nullptr; }); // wait until all evaluation queued are finished.
        evaluateCounter += inEvaluation.size();

        for(int i=0; i<inEvaluation.size(); ++i){
            NNResultBuf* buf = resultBuffer[i];
            std::vector<Node*> path = updateQueue[i];
            Node* cur = inEvaluation[i]; 

            if(buf->result == nullptr){
                std::cerr << "nullptr exception! " << i << " " << inEvaluation.size() << std::endl;
                for(int j=0; j<inEvaluation.size(); ++j){
                    std::cerr << resultBuffer[j]->result << std::endl; 
                }
            }
            std::vector<float> evalP = std::get<0>(*(buf->result));

            cur->edgeP = softmax(evalP, cur->available_moves);
            cur->edgeN = std::vector<float>(cur->edgeP.size(), 0.0f);
            std::tie(cur->initQ, cur->initW) = calculateQ(std::get<1>(*(buf->result)), std::get<3>(*(buf->result)));
            cur->initS = std::get<2>(*(buf->result));

            float evalQ = cur->initQ;

            // BACKUP (revert VL + add value)
            for (int j = path.size() - 1; j >= 0; --j) {
                Node* n = path[j];
                n->W += 1.0f;   // revert VL
                n->W += evalQ;
                evalQ = -evalQ;
            }
        }

        for(int i=0; i<resultBuffer.size(); ++i)
            delete resultBuffer[i];

        resultBuffer.clear();
        updateQueue.clear();
        inEvaluation.clear();
        searchStuck = false;
    }
}
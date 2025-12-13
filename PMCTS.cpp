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
thread_local size_t terminalHit = 0;

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
    stats[7] = terminalHit;
    stats[8] = evalCacheInsertTime;
    stats[9] = evalCacheFindTime;
    return stats;
}

void MCTS::resetTimeStats(){
    expandTime = evaluateTime = searchTime = copyTime = makeMoveTime = extraTime = 
    evalCacheHit = terminalHit = evalCacheInsertTime = evalCacheFindTime = 0;
}
#endif

const Hash hash;

std::vector<float> Node::softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves){
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
Node::Node(const Game& g, const HashValue hashValue, std::unordered_map<HashValue, Node*>* const trans_table):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), P(0.0f), initQ(0.0f), winmove(resignMove), hashValue(hashValue), trans_table(trans_table){
}

void Node::addChild(int r, int c, Game ng){
    HashValue newHash = hash.computeHashAfterMove(game, {r, c}, hashValue);
    Node* childNode;

    #ifdef transTable
    if(trans_table->count(newHash) == 0){
        childNode = new Node(ng, newHash, trans_table);
        (*trans_table)[newHash] = childNode;
    }
    else{
        childNode = (*trans_table)[newHash];
    }
    #endif
    #ifndef transTable
    childNode = new Node(ng, newHash, trans_table);
    #endif
    child.push_back(childNode);
    available_moves.push_back({r, c});
}

void Node::expand(){
    #ifdef measureTime
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    #endif

    color clr;
    std::vector<Game> nextGames(boardSize + 1, game); // +1 for pass
    std::bitset<boardSize + 1> candidateLegal; // mark candidate legal moves

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
        nextGames[boardSize].makeMoveNoScoreUpdate(passMove);
        candidateLegal[boardSize] = true;
    }

    if(game.getMoveCount() >= 2){ // if after second move, update scores & remove useless moves
        for(size_t idx = 0; idx < candidateLegal.size(); ++idx){
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
    for(size_t idx = 0; idx < candidateLegal.size(); ++idx){
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

float Node::searchandPropagate(Evaluator* evaluator){
    if(N++ == 0){
        expand();
        //std::cerr << available_moves.size() << " children expanded." << std::endl;
    }
    
    if(winmove != resignMove){ // position is won
        W--;
        //std::cerr << static_cast<int>(winmove.first) << "," << static_cast<int>(winmove.second) << " is winning move." << std::endl;
        #ifdef measureTime
        terminalHit++;
        #endif
        return 1.0f;
    }
    if(available_moves.size() == 0){ // position is lost
        W++;
        //std::cerr << "no available moves, lost position" << std::endl;
        #ifdef measureTime
        terminalHit++;
        #endif
        return -1.0f;
    }

    if(N == 1){
        #ifdef measureTime
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #endif

        NNResultBuf buf;
        bool cacheHit = evaluator->evaluate(buf, &game, hashValue);
        auto entry = buf.result;
        auto& logp = entry->first;
        auto q = entry->second;

        #ifdef measureTime
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        evaluateTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        evalCacheHit += cacheHit ? 1 : 0;
        #endif

        std::vector<float> p = softmax(logp, available_moves);

        #ifdef dirichletNoise
        std::vector<float> eta = sample_dirichlet(available_moves.size(), alpha);
        for(int i=0; i<available_moves.size(); ++i){
            child[i]->P = (1-eps) * p[i] + eps * eta[i];
        }
        #endif

        #ifndef dirichletNoise
        for(int i=0; i<available_moves.size(); ++i){
            child[i]->P = p[i];
        }
        #endif

        initQ = q;
        W += q;
        return -q;
    }
    

    int maxi = 0;
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

    float r = child[maxi]->searchandPropagate(evaluator); // rarely, NN eval seems to contain corrupted value.
    W += r;
    return -r;
}

Move Node::selectMove(float temp){
    if(winmove != resignMove)
        return winmove;
    if(available_moves.size() == 0){ // if lost, resign
        return resignMove;
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
        std::cout << "move : " << static_cast<int>(available_moves[i].first) << " " << static_cast<int>(available_moves[i].second) << 
        " sc: " << child[i]->N << " wc: " << 
        child[i]->W << " initQ : " << child[i]->initQ << " P " << child[i]->P << std::endl;
    }
    return available_moves[maxi];
}

MoveData Node::selectMoveProb(float temp){
    std::array<float, outputSize> visitPortion;
    visitPortion.fill(0.0f);

    if(winmove != resignMove)
        return {winmove, visitPortion};
    if(available_moves.size() == 0){ // if lost, resign
        return {resignMove, visitPortion};
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
    game.displayBoardGUI();
    std::cout << std::endl;
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


MCTS::MCTS(int playout, Evaluator* evaluator) : 
playout(playout), evaluator(evaluator), trans_table(new std::unordered_map<HashValue, Node*>()){
    root = new Node(Game(), hash.baseHash(), trans_table);
    #ifdef transTable
    (*trans_table)[hash.baseHash()] = root;
    #endif
}

MCTS::MCTS(MCTS&& other) noexcept
    : root(other.root), playout(other.playout),
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
    for(int i=0; i<playout; ++i){
        //std::cerr << "on playout " << i << std::endl;
        root->searchandPropagate(evaluator);
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
    #ifdef transTable
    for (auto& [hash, node] : *trans_table) {
        delete node;
    }
    trans_table->clear();
    #endif
    #ifndef transTable
    root->deleteTree();
    #endif
    root = new Node(Game(), hash.baseHash(), trans_table);
    #ifdef transTable
    (*trans_table)[hash.baseHash()] = root;
    #endif
}
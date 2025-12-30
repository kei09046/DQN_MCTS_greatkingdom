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

std::vector<std::atomic<float>> Node::softmax(const std::vector<float>& logit, const std::vector<Move>& available_moves){
    std::vector<float> n_logit;
    int size = available_moves.size();

    n_logit.reserve(size);
    for(const auto& move : available_moves){
        n_logit.push_back(logit[move.first * colSize + move.second]);
    }
    float max_logit = *std::max_element(n_logit.begin(), n_logit.end()); // For numerical stability

    // Compute exponentials after subtracting max_logit
    std::vector<float> exp_logit(size);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        exp_logit[i] = std::exp(n_logit[i] - max_logit);
        sum_exp += exp_logit[i];
    }

    // Normalize
    for (auto& val : exp_logit) {
        val /= sum_exp;
    }

    std::vector<std::atomic<float>> ret(size);
    for(size_t i=0; i<size; ++i)
        ret[i].store(exp_logit[i]);

    return ret;
}

// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, TransTable* const trans_table, Evaluator* evaluator):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), winmove(resignMove), hashValue(hashValue), trans_table(trans_table),
 evaluator(evaluator), state(NodeState_::NEEDEXPAND){
}

void Node::addChild(int r, int c, Game ng){
    HashValue newHash = hash.computeHashAfterMove(game, {r, c}, hashValue);
    Node* childNode;

    #ifdef transTable
    if(!trans_table->get(newHash, childNode)){
        childNode = new Node(ng, newHash, trans_table, evaluator);
        trans_table->insert(newHash, childNode);
        //(*trans_table)[newHash] = childNode;
    }
    #endif
    #ifndef transTable
    childNode = new Node(ng, newHash, trans_table, evaluator);
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

float Node::evaluate(){
    NNResultBuf buf;
    bool cacheHit = evaluator->evaluate(buf, &game, hashValue); // evaluation phase, set p q
    auto entry = buf.result;
    auto& logp = entry->first;
    auto q = entry->second;

    edgeP = softmax(logp, available_moves);
    edgeN = std::vector<std::atomic<float>>(edgeP.size());
    for(auto& v : edgeN)
        v.store(0.0f);

    initQ = q;
    W.fetch_add(q);
    N.fetch_add(1.0f);
    return q;
}

float Node::searchandPropagate(){
    NodeState_ current_state = state.load(std::memory_order_acquire); // thread-wise copy of current state

    if(current_state == NodeState_::NEEDEXPAND){
        bool suc = state.compare_exchange_strong(current_state, NodeState_::EXPANDING, std::memory_order_acq_rel);
        if(!suc){ // other thread is here and handling the work. From here on, it's guaranteed that only one thread is handling this part.
            return errorReturn; // don't count
        }
        expand(); // expansion phase, assign children for each possible move. Modifies available_move and winmove.
        state.store(NodeState_::NEEDEVAL, std::memory_order_release);
        current_state = NodeState_::NEEDEVAL;
    }
    else if(current_state == NodeState_::EXPANDING){
        return errorReturn;
    }

    // terminal states
    if(winmove != resignMove){ // position is won
        N.fetch_add(1.0f);
        W.fetch_add(-1.0f);
        return 1.0f;
    }
    if(available_moves.size() == 0){ // position is lost
        N.fetch_add(1.0f);
        W.fetch_add(1.0f);
        return -1.0f;
    }

    // evaluation phase; TODO : make it asynchronous.
    if(current_state == NodeState_::NEEDEVAL){
        bool suc = state.compare_exchange_strong(current_state, NodeState_::EVALUATING, std::memory_order_acq_rel);
        if(!suc){ // other thread is here and handling the work. From here on, it's guaranteed that only one thread is handling this part.
            return errorReturn; // don't count
        }
        
        float q = evaluate(); // handles internal state update as well.
        state.store(NodeState_::FINAL, std::memory_order_release);
        return -q;
    }
    else if(current_state == NodeState_::EVALUATING){
        return errorReturn;
    }

    // selection phase;
    // in non terminal case, pick move based on cPUCT formula. 
    // Unlike training, on play, FPU seems to be okay.
    assert(current_state == NodeState_::FINAL);

    int maxi = 0;
    float pref, maxval = -1.0f;
    for(int i=0; i<available_moves.size(); ++i){
        pref = ((edgeN[i].load() == 0.0f) ? -(W.load()/N.load()) : child[i]->W.load() / child[i]->N.load()) +
         cPuct * edgeP[i].load() * sqrt(N.load())/(1 + edgeN[i].load());
        
        if(maxval < pref){
            maxval = pref; 
            maxi = i;
        }
    }

    //apply virtual loss
    child[maxi]->N.fetch_add(1.0f);
    child[maxi]->W.fetch_add(-1.0f);

    float r = child[maxi]->searchandPropagate();

    //revert virtual loss
    child[maxi]->N.fetch_add(-1.0f);
    child[maxi]->W.fetch_add(1.0f);

    // backprop phase
    if(r == errorReturn){
        return r;
    }
    else{
        N.fetch_add(1.0f);
        edgeN[maxi].fetch_add(1.0f);
        W.fetch_add(r);
        return -r;
    }
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

    for(int i=0; i<available_moves.size(); ++i){
        std::cerr << "move : " << static_cast<int>(available_moves[i].first) << " " << static_cast<int>(available_moves[i].second) << 
        " sc: " << edgeN[i] << " Q: " << 
        child[i]->W/child[i]->N << " initQ : " << child[i]->initQ << " P " << edgeP[i] << std::endl;
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
        N.fetch_add(1.0f);
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

#ifdef dirichletNoise 
void Node::addDirichletNoise(){ // have to make sure that dirichlet noise is not added to same node twice. In greatKingdom, there's no repetition.
    // Have to make sure that expansion and evaluation are done before adding dirichlet noise
    assert(state == NodeState_::NEEDEXPAND || state == NodeState_::NEEDEVAL || state == NodeState_::FINAL);

    if(state == NodeState_::NEEDEXPAND){
        expand();
    }
    if(winmove != resignMove || available_moves.size() == 0) // if terminal state
        return;

    if(state == NodeState_::NEEDEVAL || state == NodeState_::NEEDEXPAND){
        evaluate();
    }
    state = NodeState_::FINAL;
    std::vector<float> eta = sample_dirichlet(edgeP.size(), alpha); 
    for(int i=0; i<edgeP.size(); ++i)
        edgeP[i] = (1-eps) * edgeP[i] + eps * eta[i];
}
#endif


MCTS::MCTS(int playout, Evaluator* evaluator) : 
playout(playout), evaluator(evaluator), trans_table(new TransTable()), thread_pool(search_thread_num){
    root = new Node(Game(), hash.baseHash(), trans_table, evaluator);
    #ifdef transTable
    trans_table->insert(hash.baseHash(), root);
    //(*trans_table)[hash.baseHash()] = root;
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
    thread_pool.join();
}

void MCTS::runSimulation(){
    #ifdef dirichletNoise
    // expands/evaluates if needed
    root->addDirichletNoise();
    #endif

    if(search_thread_num > 1){
        std::latch done(search_thread_num);
        for(int j=0; j<search_thread_num; ++j){
            boost::asio::post(thread_pool, [&, j]{
                int cnt = 0;
                while(cnt < playout/search_thread_num){
                    bool suc = (root->searchandPropagate() != errorReturn);
                    if(suc) cnt++;
                }
                done.count_down();
            });
        }
        done.wait();
    }
    else{ // no apv-mcts
        for(int i=0; i<playout; ++i)
            root->searchandPropagate();
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
    // for (auto& [hash, node] : *trans_table) {
    //     delete node;
    // }
    // trans_table->clear();
    delete trans_table;
    trans_table = new TransTable();
    #endif
    #ifndef transTable
    root->deleteTree();
    #endif
    root = new Node(Game(), hash.baseHash(), trans_table, evaluator);
    #ifdef transTable
    trans_table->insert(hash.baseHash(), root);
    //(*trans_table)[hash.baseHash()] = root;
    #endif
}
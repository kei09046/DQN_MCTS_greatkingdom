#include "PMCTS.h"
#include "analysis.hpp" // for the Analysis* hot-path hook -- see MCTS::attachAnalysis in PMCTS.h
#include "gamerules.h"
#include "neuralNet.h"
#include "random.h"
#include "hash.h"
#include "evaluator.h"
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <chrono>
#include <ranges>
#include <execution>



namespace{
    const Hash hash;
}

// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, TransTable* const transposTable):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), S(0.0f), Wp(0.0f), minN(0.0f), forcedState(0), hashValue(hashValue), expanded(false), transposTable(transposTable){
}

void Node::addChild(const Move& move, int idx){
    HashValue newHash = hash.computeHashAfterMove(game, move, hashValue);

    Node* childNode;
    Color winner;

    if(globalConfig.transTable){
        auto it = transposTable->find(newHash);

        if(it == transposTable->end()){
            Game ng = this->game;

            // if idx = -1, move is not in the list. -> Add completely new child.
            if(idx == -1){
                winner = ng.makeMove(move).first;
            }
            // else, compute child based on stats calculated on setPolicyMask() call.
            else{
                winner = ng.makeMoveGivenScore(move);
            }
            // reset policyMask state. It would be reevaluated when this node gets selected during search.
            ng.resetMask();
            childNode = new Node(ng, newHash, transposTable);
            childNode->forcedState = (winner == EMPTY) ? 0 : (winner == turn) ? -1 : 1;
            transposTable->emplace(newHash, std::make_pair(childNode, 1));
        }
        else{
            childNode = it->second.first;
            (it->second.second)++;
        }
    }
    else{
        Game ng = this->game;
        if(idx == -1){
            winner = ng.makeMove(move).first;
        }
        else{
            winner = ng.makeMoveGivenScore(move);
        }
        childNode = new Node(ng, newHash, transposTable);
        childNode->forcedState = (winner == EMPTY) ? 0 : (winner == turn) ? -1 : 1;
    }

    if(idx == -1)
        child.push_back(childNode);
    else
        child[idx] = childNode;
}



void Node::expand(){
    //game.displayBoardGUI(true);
    //std::cerr << "Expand called" << std::endl;

    expanded = true;
    // if forcedState != 0, then there is either a winning move or there will be no available moves anyway. Don't expand.
    if(forcedState != 0)
        return;

    child = std::vector<Node*>(game.getAvailableMoves().size(), nullptr);
    
    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

int Node::selectChildInSearch(float fpu){
    int maxi = -1;
    float pref, maxval = -1000.0f; // pref may be less than -1.(due to score head)
    bool lost = true;

    if(game.getAvailableMoves().empty()){
        std::cerr << N << " " << W << " " << initQ << " " << S << " " << Wp << std::endl;
        game.displayBoardGUI(true);
    }
    assert(!game.getAvailableMoves().empty());

    // KataGo style fpu.
    float fpuValue = 0.0f;
    if(fpu >= 0.0f){
        float visitedPolicy = 0.0f;
        for(int i=0; i<edgeN.size(); ++i)
            if(edgeN[i] > 0.0f)
                visitedPolicy += edgeP[i];
        fpuValue = -W/valueN() - fpu * std::sqrt(visitedPolicy);
    }

    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        if(child[i] == nullptr){
            pref = fpuValue + globalConfig.cPuct * edgeP[i] * sqrt(N);
            lost = false;
        }

        else{
            int forced = child[i]->forcedState;
            
            // if winning continuation found.
            if(forced < 0){
                forcedState = -forced + 1;
                return i;
            }
            // only select non-losing move.
            if(forced == 0){
                pref = child[i]->W / child[i]->N + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
                lost = false;
            }
            // losing move will almost not be selected if there is a non-losing move.
            else{ 
                pref = -2.0f + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
            }
        }
            
        if(maxval < pref){
            maxval = pref; 
            maxi = i;
        }
    }
    
    // if every move is lost, update forcedState. Move with highest P value would be selected.
    if(lost){
        forcedState = -child[maxi]->forcedState - 1;
    }

    // Allocate child only when child gets selected; Delay memory allocation as much as possible.
    if(child[maxi] == nullptr)
        addChild({game.getAvailableMoves()[maxi] / colSize, game.getAvailableMoves()[maxi] % colSize}, maxi);
    return maxi;
}

Move Node::selectMove(float temp, int minVisit){
    //std::cout << "available move size : " << game.getAvailableMoves().size() << std::endl;
    if(forcedState == -1 || forcedState == 1){
        game.displayBoardGUI(true);
        assert(false && "trying to make move on terminated position!");
    }

    else if(forcedState > 0){
        // if(globalConfig.detailedStat)
        //     std::cout << "status: " << static_cast<int>(onlyMove.first) << " " << static_cast<int>(onlyMove.second)
        //     << " forced : " << -forcedState + (forcedState > 0 ? 1 : -1) << std::endl;
        if(game.getWin() != RESIGNMOVE){
            return game.getWin();
        }
        for(int i=0; i<child.size(); ++i){
            if(child[i] != nullptr && child[i]->forcedState < 0)
                return {game.getAvailableMoves()[i] / colSize, game.getAvailableMoves()[i] % colSize};
        }
        assert(false && "no losing child found despite winning! = no winning move found!");
    }
    else if(game.getAvailableMoves().empty()){
        return RESIGNMOVE;
    }


    int maxi, maxn = -1, index;
    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
    }

    // proven lost root (forcedState < -1) only gets here with every move losing: just take the most visited one.
    if(temp >= 5.0f || game.getMoveCount() >= 10 || forcedState < 0){
        return {game.getAvailableMoves()[maxi] / colSize, game.getAvailableMoves()[maxi] % colSize};
    }

    std::vector<float> weights(game.getAvailableMoves().size());
    std::vector<float> cumulative(game.getAvailableMoves().size());
    
    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        // proven losing moves get no minimum visits (see MCTS::runSimulation) and are never picked.
        if(child[i] != nullptr && child[i]->forcedState > 0){
            weights[i] = 0.0f;
            continue;
        }
        weights[i] = (edgeN[i] - minVisit <= 0) ? 0.0f : std::pow(edgeN[i] - minVisit, temp);
    }
    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());
    // no move above minVisit (e.g. a short search) : take the most visited one.
    if(cumulative.back() <= 0.0f)
        return {game.getAvailableMoves()[maxi] / colSize, game.getAvailableMoves()[maxi] % colSize};

    std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
    float rnd = dist(rnd::gen);

    auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
    index = std::distance(cumulative.begin(), it);
    return {game.getAvailableMoves()[index] / colSize, game.getAvailableMoves()[index] % colSize};

}


MoveData Node::selectMoveWithStat(float temp, int minVisit){
    std::vector<float> visitPortion(outputSize, 0.0f);
    Move selectedMove;

    const auto& moves = game.getAvailableMoves();
    assert(!moves.empty());

    auto targetWeights = [&](){
        std::vector<float> w(moves.size(), 0.0f);
        for(int i=0; i<moves.size(); ++i)
            w[i] = std::max(edgeN[i] - minVisit, 0.0f);
        if(std::accumulate(w.begin(), w.end(), 0.0f) > 0.0f)
            return w;

        for(int i=0; i<moves.size(); ++i)
            w[i] = edgeN[i];
        if(std::accumulate(w.begin(), w.end(), 0.0f) > 0.0f)
            return w;

        std::fill(w.begin(), w.end(), 1.0f);
        return w;
    };

    // writes the normalized weights into visitPortion.
    auto setVisitPortion = [&](const std::vector<float>& w){
        const float sum = std::accumulate(w.begin(), w.end(), 0.0f);
        for(int i=0; i<moves.size(); ++i)
            visitPortion[moves[i]] = w[i] / sum;
    };

    if(forcedState == -1 || forcedState == 1){
        game.displayBoardGUI(true);
        assert(false && "trying to make move on terminated position!");
    }

    if(forcedState > 0){
        // apply label smoothing.
        //std::fill(visitPortion.begin(), visitPortion.end(), 0.02f/(outputSize - 1));

        if(game.getWin() != RESIGNMOVE){
            selectedMove = game.getWin();
            visitPortion[selectedMove.first * colSize + selectedMove.second] = 1.0f;
        }
        else{
            for(int i=0; i<child.size(); ++i){
                if(child[i] != nullptr && child[i]->forcedState < 0){
                    auto winningMove = game.getAvailableMoves()[i];
                    visitPortion[winningMove] = 1.0f;
                    selectedMove = {winningMove / colSize, winningMove % colSize};
                    break;
                }
            }
        }
    }
    // select most visited move. 
    else if(temp >= 5.0f || game.getMoveCount() >= 10 || forcedState < 0){
        const std::vector<float> w = targetWeights();
        setVisitPortion(w);
        const int maxi = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
        selectedMove = {moves.at(maxi) / colSize, moves.at(maxi) % colSize};
    }

    // root is not determined win/loss after playouts.
    else{
        const std::vector<float> w = targetWeights();
        setVisitPortion(w);

        std::vector<float> cumulative(moves.size()), weights(moves.size());
        for(int i=0; i<moves.size(); ++i)
            weights[i] = (w[i] <= 0.0f) ? 0.0f : std::pow(w[i], temp);
        std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(rnd::gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        int index = std::distance(cumulative.begin(), it);
        selectedMove = {moves.at(index) / colSize, moves.at(index) % colSize};
    }

    // std::cerr << "Available moves : ";
    // for(const auto& m : game.getAvailableMoves())
    //     std::cerr << m/colSize << m%colSize << " ";
    // std::cerr << std::endl;
    // std::cerr << "Selected moves : ";
    // printMove(selectedMove);

    std::optional<float> value;
    if(forcedState > 0)
        value = -0.9f;
    else if(forcedState < 0)
        value = 0.9f;
    else if(valueN() > 0.0f)
        value = std::clamp(Wp / valueN(), -0.9f, 0.9f);

    return {selectedMove, visitPortion, forcedState, game.getAvailableMoves().size() == 1, value};
}

Node* Node::jump(Move move){
    if(N == 0){
        game.setPolicyMask();
    }
    if(!expanded){
        expand();
    }
    // a proven node (e.g. an immediate win found on its first visit) is never expanded, so it has no child slots yet.
    if(child.size() < game.getAvailableMoves().size())
        child.resize(game.getAvailableMoves().size(), nullptr);
    N++;

    // std::cerr << "requested move : " << static_cast<int>(move.first) << "," << static_cast<int>(move.second) << std::endl;
    // std::cerr << "available options : " << std::endl;
    // for(auto p : game.getAvailableMoves())
    //     std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";
    // std::cerr << "node's state : " << std::endl;
    // game.displayBoardGUI(true);

    int moveInt = move.first * colSize + move.second;
    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        if(game.getAvailableMoves()[i] == moveInt){
            // delay child allocation as much as possible. Happens when agent is forced to jump to a move that it hasn't considered at all.
            if(child[i] == nullptr){
                addChild(move, i);
            }
            return child[i];
        }
    }

    std::cerr << "unexpected move played" << std::endl;
    addChild(move, -1);
    return child.back();
}

void Node::deleteTree(){
    if(!globalConfig.transTable){
        for(Node* c : child){
            if(c != nullptr)
                c->deleteTree();
        }
        delete this;
        return;
    }

    auto it = transposTable->find(hashValue);
    // if(it == transposTable->end()){
    //     std::cerr << "hash : " << hashValue << std::endl;
    //     this->game.displayBoardGUI(false);
    // }
    assert(it != transposTable->end());

    if (--(it->second.second) == 0) {
        for(Node* c : child){
            if(c != nullptr)
                c->deleteTree();
        }
        // std::cerr << "deleted : " << hashValue << std::endl;
        transposTable->erase(it);
        delete this;
        return;
    }
}

void Node::deleteTree(Node* exception){
    if(!globalConfig.transTable){
        for(Node* c : child){
            if(c != exception && c != nullptr)
                c->deleteTree();
        }
        delete this;
        return;
    }

    auto it = transposTable->find(hashValue);
    if(it == transposTable->end()){ // should never enter here.
        std::cerr << "hash : " << hashValue << std::endl;
        this->game.displayBoardGUI(false);
    }
    assert(it != transposTable->end());

    if (--(it->second.second) == 0) {
        for(Node* c : child){
            if(c != exception && c != nullptr)
                c->deleteTree();
        }
        // std::cerr << "deleted : " << hashValue << std::endl;
        transposTable->erase(it);
        delete this;
        return;
    }
}

void Node::addDirichletNoise(Evaluator* evaluator){
    if(N == 0){
        game.setPolicyMask();
        N++;
    }
    if (!expanded) {
        expand();
        if(forcedState == 0){
            auto buf = std::make_shared<NNResultBuf>();
            evaluator->evaluate(buf, &game, hashValue);

            edgeP = buf->result->P;
            edgeN.assign(edgeP.size(), 0.0f);
        }
    }

    // Dirichlet noise is only applied when position is undetermined or lost.
    if(forcedState <= 0 && game.getAvailableMoves().size() > 0){
        std::vector<float> eta = rnd::sample_dirichlet(edgeP.size(), globalConfig.alpha); 
        for(int i=0; i<edgeP.size(); ++i)
            edgeP[i] = (1-globalConfig.eps) * edgeP[i] + globalConfig.eps * eta[i];
    }
}

MCTS::MCTS(Evaluator* evaluator, SearchParams params) : 
evaluator(evaluator), transposTable(new TransTable()), params(params){
    root = new Node(Game(), hash.baseHash(), transposTable);
    if(globalConfig.transTable){
        transposTable->emplace(hash.baseHash(), std::make_pair(root, 1));
    }
}

MCTS::MCTS(MCTS&& other) noexcept
    : root(other.root), evaluator(other.evaluator), transposTable(other.transposTable), params(other.params)
{
    other.root = nullptr;
    other.evaluator = nullptr;
    other.transposTable = nullptr;
}

MCTS::~MCTS(){
    delete transposTable;
}

void MCTS::runSimulation(const int playMode, const int nPlayout, const int timeLimit, const int minSearchPerChild, 
    const bool fastSearch, const bool earlyQuit){
    // root already proven won/lost: nothing to search.
    if(root->forcedState != 0)
        return;

    if(!fastSearch && globalConfig.dirichletNoise)
        root->addDirichletNoise(evaluator);

    int search_counter = 0;
    int evaluate_counter = 0;
    std::vector<Node*> current_evaluating_nodes;
    std::vector<std::vector<Node*>> need_update_chain;
    std::vector<std::shared_ptr<NNResultBuf>> result_buffer;
    bool stuck_during_search = false; // happens if meet evaluating node while searching

    auto flushPendingEval = [&](){
        if(!current_evaluating_nodes.empty()){
            std::shared_ptr<NNResultBuf> rb = result_buffer.back();
            std::unique_lock<std::mutex> lk2(rb->resultmutex);
            rb->resultcv.wait(lk2, [&]{ return rb->result != nullptr; });

            for(size_t i=0; i<current_evaluating_nodes.size(); ++i){
                updateEval(result_buffer[i], need_update_chain[i], current_evaluating_nodes[i]);
            }
            evaluate_counter += current_evaluating_nodes.size();
            result_buffer.clear();
            need_update_chain.clear();
            current_evaluating_nodes.clear();
        }
        stuck_during_search = false;
    };

    if(playMode == PLAYOUT){
        // step 0. expand root.
        while(!root->expanded && root->forcedState == 0)
            playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
                playMode, timeLimit);

        // step 0.5. If earlyQuit is set to true, check if |N most visited edge| - |N second most visited edge| > nPlayout and quit early.
        if(earlyQuit){
            float fv = 0.0f, sv = 0.0f;
            for(const auto v : root->edgeN){
                if(v >= fv){
                    sv = fv;
                    fv = v;
                }
                else if(v > sv){
                    sv = v;
                }
            }
            if(fv - sv > nPlayout)
                return;
        } 

        // step 1 : force minSearchPerChild visit for all children.
        if(minSearchPerChild > 0 && root->forcedState == 0){
            for(int i=0; i<minSearchPerChild && root->forcedState == 0; ++i){
                for(int j=0; j<root->child.size(); ++j){
                    auto c = root->child[j];
                    
                    // if child is undetermined.
                    if(c == nullptr || c->forcedState == 0){
                        playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
                        playMode, timeLimit, j);
                    }
                    // if child is proven lost. -> root is proven won.
                    else if(c != nullptr && c->forcedState < 0){
                        root->forcedState = -c->forcedState + 1;
                        break;
                    }
                }
                flushPendingEval();
            }
        }

        // step 2. continue searching normally.
        while(search_counter < nPlayout && (root->forcedState == 0)){
            playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
            playMode, timeLimit);
        }
    }
    else{
        auto duration = std::chrono::seconds(timeLimit);
        auto start = std::chrono::steady_clock::now();
        while(std::chrono::steady_clock::now() - start < duration && root->forcedState == 0){
            playout(search_counter, evaluate_counter, current_evaluating_nodes, need_update_chain, result_buffer, stuck_during_search,
            playMode, nPlayout, timeLimit);
        }
    }

    // flush pending evaluations.
    flushPendingEval();

    // if(globalConfig.detailedStat)
    //     std::cout << "playout : " << search_counter << " " << evaluate_counter << std::endl;
}

Move MCTS::getMove(float temp){
    const int minVisit = params.minVisits(globalConfig.nPlayout);
    for(int i=0; i<10; ++i){
        runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout / 10, globalConfig.time / 10,
            minVisit * (i + 1) / 10 - minVisit * i / 10);
        // printVariation();
        const auto& [winProb, scoreEXP] = getEval();
        // std::cout << "winprob : " << winProb << "\nscoreEXP : " << scoreEXP << std::endl;
    }
    return selectMove(temp);
}

Move MCTS::selectMove(float temp){
    return root->selectMove(temp, params.minVisits(globalConfig.nPlayout));
}

MoveData MCTS::getMoveWithStat(float temp){
    const int minVisit = params.minVisits(globalConfig.nPlayout);
    runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout, globalConfig.time, minVisit, false, true);
    return root->selectMoveWithStat(temp, minVisit);
}

Move MCTS::getFastMove(float temp){
    runSimulation(PLAYOUT, globalConfig.fastPlayout, globalConfig.time, -1, true, true);
    return root->selectMove(temp, 0);
}

std::pair<float, float> MCTS::getEval(){
    assert(root->N > 0);
    if(root->forcedState == 0)
        return {static_cast<float>(-root->W) / root->valueN(), static_cast<float>(-root->S) / root->valueN()};
    else if(root->forcedState > 0)
        return {1.0f, 0.0f};
    else
        return {-1.0f, 0.0f};
}

bool MCTS::jump(Move move){
    Node* old_root = root;
    root = root->jump(move);
    old_root->deleteTree(root);
    return root != nullptr;
}

void MCTS::reset(const Game& startPos){
    root->deleteTree();
    root = new Node(startPos, hash.computeHash(startPos), transposTable);
    root->game.resetMask();

    if(globalConfig.transTable){
        transposTable->clear();
        transposTable->emplace(hash.computeHash(startPos), std::make_pair(root, 1));
    }
}

void MCTS::playout(int& searchCounter, int& evaluateCounter,
    std::vector<Node*>& inEvaluation, std::vector<std::vector<Node*>>& updateQueue,
    std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int timeLimit, int forcedFirstSearch) {
    // std::cerr << searchCounter << " " << evaluateCounter << std::endl;

    // SELECTION
    if((inEvaluation.size() < globalConfig.search_thread_num) && !searchStuck){
        std::vector<int> childIdx;
        std::vector<Node*> path;
        Node* cur = root;
        float evalQ = 0.0f, evalS = 0.0f, evalW = 0.0f;
        int forced = 0; // check if state is forced win / loss

        while (true) {
            path.push_back(cur);

            // on first visit, set up policyMask then just quit.
            if(cur->N == 0.0f){
                cur->game.setPolicyMask();
                if(cur->game.getAvailableMoves().empty()){
                    cur->forcedState = -2;
                }
                if(cur->game.getWin() != RESIGNMOVE){
                    cur->forcedState = 2;
                }
                forced = cur->forcedState;
                break;
            }

            // if node is evaluating, return
            if(std::find(inEvaluation.begin(), inEvaluation.end(), cur) != inEvaluation.end()){
                searchStuck = true;
                return;
            }

            // on second visit, do expansion
            if (!(cur->expanded)) {
                assert(cur->N == 1);
                cur->expand();
                forced = cur->forcedState;
                if(forced == 0 && cur->edgeN.empty()){
                    cur->edgeN.assign(cur->edgeP.size(), 0.0f);
                }
            }

            forced = cur->forcedState;
            if(forced != 0)
                break;

            int a;
            if(cur == root && forcedFirstSearch >= 0){
                a = forcedFirstSearch;
                // selectChildInSearch (the ordinary path below) allocates child[a] itself,
                // lazily, the instant before returning it -- this bypasses that entirely, so it
                // has to do the same allocate-if-missing check here, or an unexplored index
                // (child[a] still nullptr, since root->child is only ever pre-sized with
                // placeholders in Node::expand(), not populated) segfaults on the dereference
                // below.
                if(cur->child[a] == nullptr){
                    const auto& moves = cur->game.getAvailableMoves();
                    cur->addChild({moves[a] / colSize, moves[a] % colSize}, a);
                }
            }
            else
                a = cur->selectChildInSearch(params.fpu); // assume node is evaluated

            childIdx.push_back(a);
            cur = cur->child[a];
        }

        searchCounter++;
        // set node visit stats
        for(int i=0; i<childIdx.size(); ++i){
            path[i]->edgeN[childIdx[i]] += 1.0f;
        }

        // Minimum-visit playouts must not affect root's Q, or forced looks at bad moves (e.g. ones
        // that only set up traps) leak into root's value. Root still counts the visit in N (so N
        // stays consistent with edgeN), but it is tracked in minN and dropped from the path, so it
        // gets no VL and propagate never adds to its W/S/Wp. Use valueN() to read root's Q.
        if(forcedFirstSearch >= 0 && !childIdx.empty()){
            root->N += 1.0f;
            root->minN += 1.0f;
            path.erase(path.begin());
        }

        for (Node* node : path) {
            node->N += 1.0f;
            node->W -= 1.0f; // apply VL
        }

        if(forced != 0){ // if forced win/loss is found in leaf node, propagate that result immediately.
            evaluateCounter++;
            //propagate(path, childIdx, forced);

            if(forced > 0)
                propagate(path, -1.0f, -1.0f, -cur->game.scoreDiff(cur->turn));
            else
                propagate(path, 1.0f, 1.0f, -cur->game.scoreDiff(cur->turn));

            if(analysis && analysis->getDebugMode())
                analysis->printPlayoutDebugLine(debugPath(path), 0.0f, 0.0f, forced);
        }
        else{ // if final search node is non-determined node, ask for evaluation
            // enqueue evaluation
            auto buf = std::make_shared<NNResultBuf>();

            bool cacheHit = evaluator->asyncEvaluate(buf, &(cur->game), cur->hashValue); // check cacheHit, also request eval
            if(!cacheHit){
                resultBuffer.push_back(buf);
                inEvaluation.push_back(cur);
                // std::cout << "query evaluation : " << cur << std::endl;
                updateQueue.push_back(path);
            }
            else{
                // if eval is cached, do param update right away.
                updateEval(buf, path, cur);
                evaluateCounter++;
            }
        }
    }

    //EVALUATION & UPDATE
    if(inEvaluation.size() >= globalConfig.search_thread_num || searchStuck){
        // wait for the result
        std::shared_ptr<NNResultBuf> rb = resultBuffer.at(inEvaluation.size() - 1);
        std::unique_lock<std::mutex> lk2(rb->resultmutex);
        rb->resultcv.wait(lk2, [&]{ return rb->result != nullptr; }); // wait until all evaluation queued are finished.

        for(int i=0; i<inEvaluation.size(); ++i){
            std::shared_ptr<NNResultBuf> buf = resultBuffer[i];
            std::vector<Node*> path = updateQueue[i];
            Node* cur = inEvaluation[i];

            updateEval(buf, path, cur);
        }

        evaluateCounter += inEvaluation.size();
        resultBuffer.clear();
        updateQueue.clear();
        inEvaluation.clear();
        searchStuck = false;
    }
}

void MCTS::updateEval(const std::shared_ptr<NNResultBuf> buf, const std::vector<Node*> path, Node* cur){
    float evalS, evalW, evalQ;

    // std::cout << "get evaluation : " << cur << " " << buf->result << " " << cur->evaluation << std::endl;

    // S and Wp are always backed up: Wp (mean win probability) is the root value used for value targets in self-play.
    auto r = buf->result;
    cur->initQ = r->U;
    cur->edgeP = r->P;
    evalW = r->V;
    evalS = r->S;
    evalQ = r->U;

    if(analysis && analysis->getDebugMode())
        analysis->printPlayoutDebugLine(debugPath(path), evalW, evalS, 0);

    propagate(path, evalQ, evalW, evalS);
}


std::vector<Node*> MCTS::debugPath(const std::vector<Node*>& path) const{
    // minimum-visit playouts have root dropped from their path (see playout); put it back so the
    // printed move sequence still starts from root.
    if(path.front() == root)
        return path;
    std::vector<Node*> full{root};
    full.insert(full.end(), path.begin(), path.end());
    return full;
}

void MCTS::propagate(const std::vector<Node*>& path, float evalQ, float evalW, float evalS){
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node* n = *it;
        n->W += 1.0f;   // revert VL
        n->W += evalQ;
        n->Wp += evalW;
        n->S += evalS;
        evalQ = -evalQ;
        evalS = -evalS;
        evalW = -evalW;
    }
}
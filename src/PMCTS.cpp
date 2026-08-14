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

    std::vector<float> softmax(const std::vector<float>& logit, const Node* node){
        // edgeP/edgeN/child are indexed by position within getAvailableMoves(), not by raw
        // board position, so the returned vector must be compacted/reindexed the same way.
        const auto& availableMoves = node->game_().getAvailableMoves();
        const auto& transferList = node->game_().getTransferTable();
        const int moveSize = availableMoves.size();
        // available moves : [1, 2, 3, 7, 8, 9 ...]
        // transferList : [0, 0, 0, -1, -1, -1, 1, 0, 2, ...]

        std::vector<float> transferred(moveSize, std::numeric_limits<float>::lowest());
        for(int i=0; i<transferList.size(); ++i){
            if(transferList[i] != -1){
                transferred.at(transferList.at(i)) = std::max(transferred[transferList[i]], logit.at(i));
            }
        }

        // std::vector<float> n_logit(moveSize);
        // for(int i=0; i<moveSize; ++i){
        //     n_logit[i] = transferred.at(availableMoves[i]);
        // }

        std::vector<float> exp_logit(moveSize);
        float max_logit = *std::max_element(transferred.begin(), transferred.end()); // For numerical stability

        // Compute exponentials after subtracting max_logit
        float sum_exp = 0.0f;
        for (int i = 0; i < moveSize; ++i) {
            exp_logit[i] = std::exp(transferred[i] - max_logit);
            sum_exp += exp_logit[i];
        }

        // Normalize
        for (float& val : exp_logit) {
            val /= sum_exp;
        }

        // if(globalConfig.detailedStat){
        //     static int cntr = 0;
        //     if(cntr++ % 100 == 0){
        //         std::cout << "turn : " << static_cast<int>(node->game_().getTurn()) << std::endl;  
        //         ModelCompare::displayBoardGUI(true, node->game_());

        //         int t = 0;
        //         for(uint8_t i=0U; i<rowSize; ++i){
        //             for(uint8_t j=0U; j<colSize; ++j){
        //                 if(game.getAvailableMoves()[t] == Move{i, j}){
        //                     std::printf("%8.4f ", exp_logit[t++]);
        //                 }
        //                 else{
        //                     std::printf("%8.4f ", 0.0);
        //                 }
        //             }
        //             std::cout << std::endl;
        //         }
        //         t = 0;
        //         for(int i=0; i<rowSize; ++i){
        //             for(int j=0; j<colSize; ++j){
        //                 if(game.getAvailableMoves()[t] == Move{i, j}){
        //                     std::cout << maxIdx[t] / colSize << "," << maxIdx[t] % colSize << " "; 
        //                     t++;
        //                 }
        //                 else{
        //                     std::cout << rowSize << "," << colSize << " ";
        //                 }
        //             }
        //             std::cout << std::endl;
        //         }
        //     }
        // }

        return exp_logit;
    }

    // returns (U, Q) estimate
    std::pair<float, float> calculateQ(const std::shared_ptr<PolicyValueOutput> nnOutput, const Game& game){
        const auto& [logAct, winP, scoreEXP, scoreMap] = *nnOutput;

        float captureV[2] = {0.0f, 0.0f};
        const Color turn = game.getTurn();
        const Color oppturn = Game::reverseColor(turn);
        const Color turnScore = (turn == BLACK) ? BSCORE : WSCORE;
        const Color oppturnScore = (turn == BLACK) ? WSCORE : BSCORE;

        float scoreV = 2 * ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi);

        scoreV += scoreEXP;
        for(int i=0; i<boardSize; ++i){
            if(game.getBoard({i / colSize, i % colSize}) == EMPTY){
                Color owned = game.getScoreBoard({i / colSize, i % colSize});
                scoreV += (owned == oppturnScore) ? 1.0f : ((owned == turnScore) ? -1.0f : scoreMap[i]);
            }
        }

        // float scoreV = ((turn == BLACK) ? globalConfig.komi : -globalConfig.komi) + scoreEXP;

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
        // if(globalConfig.detailedStat){
        //     static int cntr = 0;
        //     if(cntr++ % 100 == 0){
        //         ModelCompare::displayBoardGUI(true, game);
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

        return {utility, winP};
    }
}

// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, TransTable* const transposTable):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), S(0.0f), Wp(0.0f), forcedState(0), hashValue(hashValue), expanded(false), evaluation(nullptr), transposTable(transposTable){
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
            // else, compute child based on stats calculated on expand() call.
            else{
                winner = ng.makeMoveGivenScore(move);
            }

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
    //ModelCompare::displayBoardGUI(true, game);
    //std::cerr << "Expand called" << std::endl;

    expanded = true;
    // if forcedState != 0, then there is either a winning move or there will be no available moves anyway. Don't expand.
    if(forcedState != 0)
        return;

    child = std::vector<Node*>(game.getAvailableMoves().size(), nullptr);

    if(game.getAvailableMoves().empty()){
        forcedState = -1;
    }
    
    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

int Node::selectChildInSearch(){
    int maxi = -1;
    float pref, maxval = -1000.0f; // pref may be less than -1.(due to score head)
    bool lost = true;

    if(game.getAvailableMoves().empty()){
        std::cerr << N << " " << W << " " << initQ << " " << S << " " << Wp << std::endl;
        ModelCompare::displayBoardGUI(true, game);
    }
    assert(!game.getAvailableMoves().empty());

    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        if(child[i] == nullptr){
            pref = ((globalConfig.fpu < 0.0f) ? 0.0f : -W/N-globalConfig.fpu) + globalConfig.cPuct * edgeP[i] * sqrt(N);
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

Move Node::selectMove(float temp){
    //std::cout << "available move size : " << game.getAvailableMoves().size() << std::endl;
    if(forcedState == -1 || forcedState == 1){
        ModelCompare::displayBoardGUI(true, game);
        assert(false && "trying to make move on terminated position!");
    }

    else if(forcedState > 0){
        // if(globalConfig.detailedStat)
        //     std::cout << "status: " << static_cast<int>(onlyMove.first) << " " << static_cast<int>(onlyMove.second)
        //     << " forced : " << -forcedState + (forcedState > 0 ? 1 : -1) << std::endl;
        for(int i=0; i<child.size(); ++i){
            if(child[i] != nullptr && child[i]->forcedState < 0)
                return {game.getAvailableMoves()[i] / colSize, game.getAvailableMoves()[i] % colSize};
        }
        assert(false && "no losing child found despite winning! = no winning move found!");
    }
    else if(game.getAvailableMoves().empty()){
        return RESIGNMOVE;
    }
    // else if(globalConfig.detailedStat){
    //     std::vector<int> v(game.getAvailableMoves().size());
    //     std::iota(v.begin(), v.end(), 0);
    //     std::sort(v.begin(), v.end(), [&](const int& a, const int& b){
    //         return edgeN[a] > edgeN[b];
    //     });

    //     for(int i=0; i<std::min(static_cast<int>(game.getAvailableMoves().size()), 3); ++i){
    //         int idx = v[i];
    //         if(child[idx] != nullptr)
    //             std::cout << "status: " << static_cast<int>(game.getAvailableMoves()[idx].first) << " " << static_cast<int>(game.getAvailableMoves()[idx].second)
    //             << " forced : " << child[idx]->forcedState << " sc: " << edgeN[idx] << " Q: " 
    //             << child[idx]->W/child[idx]->N << " initQ : " << child[idx]->initQ << " Wp : " << child[idx]->Wp/child[idx]->N 
    //             << " S : " << child[idx]->S / child[idx]->N << " P " << edgeP[idx] << std::endl;
    //     }
    // }

    int maxi, maxn = -1, index;
    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
    }

    if(temp >= 5.0f || game.getMoveCount() >= 10){
        return {game.getAvailableMoves()[maxi] / colSize, game.getAvailableMoves()[maxi] % colSize};
    }

    std::vector<float> weights(game.getAvailableMoves().size());
    std::vector<float> cumulative(game.getAvailableMoves().size());
    for(int i=0; i<game.getAvailableMoves().size(); ++i){
        weights[i] = std::pow(edgeN[i], temp);
    }
    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

    std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
    float rnd = dist(gen);

    auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
    index = std::distance(cumulative.begin(), it);
    return {game.getAvailableMoves()[index] / colSize, game.getAvailableMoves()[index] % colSize};

}


MoveData Node::selectMoveProb(float temp){
    std::vector<float> visitPortion(outputSize, 0.0f);
    Move selectedMove;

    if(forcedState == -1 || forcedState == 1){
        ModelCompare::displayBoardGUI(true, game);
        assert(false && "trying to make move on terminated position!");
    }

    if(forcedState > 0){
        // label smoothing is applied here.
        //std::fill(visitPortion.begin(), visitPortion.end(), 0.02f/(outputSize - 1));
        for(int i=0; i<child.size(); ++i){
            if(child[i] != nullptr && child[i]->forcedState < 0){
                auto winningMove = game.getAvailableMoves()[i];
                visitPortion[winningMove] = 1.0f;
                selectedMove = {winningMove / colSize, winningMove % colSize};
                break;
            }
        }
    }

    else if(temp >= 5.0f || game.getMoveCount() >= 10){
        int maxi, maxn = -1;
        for(int i=0; i<game.getAvailableMoves().size(); ++i){
            if(edgeN[i] > maxn){
                maxn = edgeN[i];
                maxi = i;
            }
            visitPortion[game.getAvailableMoves()[i]] = edgeN[i]/N;
        }
        selectedMove = {game.getAvailableMoves()[maxi] / colSize, game.getAvailableMoves()[maxi] % colSize};
    }

    else{
        std::vector<float> cumulative(game.getAvailableMoves().size()), weights(game.getAvailableMoves().size());

        for(int i=0; i<game.getAvailableMoves().size(); ++i){
            visitPortion[game.getAvailableMoves()[i]] = edgeN[i]/N;
            weights[i] = (edgeN[i] == 0.0f) ? 0.0f : std::pow(edgeN[i], temp);
        }

        std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        int index = std::distance(cumulative.begin(), it);
        selectedMove = {game.getAvailableMoves()[index] / colSize, game.getAvailableMoves()[index] % colSize};
    }

    return {selectedMove, visitPortion, forcedState, game.getAvailableMoves().size() == 1};
}

Node* Node::jump(Move move){
    if(N == 0){
        game.setPolicyMask();
    }
    if(!expanded){
        expand();
    }
    N++;

    // std::cerr << "requested move : " << static_cast<int>(move.first) << "," << static_cast<int>(move.second) << std::endl;
    // std::cerr << "available options : " << std::endl;
    // for(auto p : game.getAvailableMoves())
    //     std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";
    // std::cerr << "node's state : " << std::endl;
    // ModelCompare::displayBoardGUI(true, game);

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

    // The requested move isn't among this node's policy-restricted available moves -- e.g.
    // Game::setPolicyMask (gamerules.cpp) only ever offers PASSMOVE when the current player is
    // already ahead on score (or under its "opponent has a winning threat" branch), so a pass
    // played outside that window is a real, engine-accepted move that this node's search simply
    // never considered. It's still legal (the caller already validated it before calling jump),
    // so graft it on as a fresh child instead of failing here: returning nullptr used to silently
    // corrupt the MCTS root, which then crashed on the very next tree operation (jump/search/etc.)
    // -- reproduced via a single "pass" during analysis mode segfaulting on the following command.
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
    //     ModelCompare::displayBoardGUI(false, this->game);
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
        ModelCompare::displayBoardGUI(false, this->game);
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

            edgeP = softmax(std::get<0>(*(buf->result)), this);
            edgeN.assign(edgeP.size(), 0.0f);
        }
    }

    // Dirichlet noise is only applied when position is undetermined or lost.
    if(forcedState <= 0 && game.getAvailableMoves().size() > 0){
        std::vector<float> eta = sample_dirichlet(edgeP.size(), globalConfig.alpha); 
        for(int i=0; i<edgeP.size(); ++i)
            edgeP[i] = (1-globalConfig.eps) * edgeP[i] + globalConfig.eps * eta[i];
    }
}

MCTS::MCTS(Evaluator* evaluator) : 
evaluator(evaluator), transposTable(new TransTable()){
    root = new Node(Game(), hash.baseHash(), transposTable);
    if(globalConfig.transTable){
        transposTable->emplace(hash.baseHash(), std::make_pair(root, 1));
    }
}

MCTS::MCTS(MCTS&& other) noexcept
    : root(other.root), evaluator(other.evaluator), transposTable(other.transposTable)
{
    other.root = nullptr;
    other.evaluator = nullptr;
    other.transposTable = nullptr;
}

MCTS::~MCTS(){
    delete transposTable;
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
        while(evaluate_counter < nPlayout && (root->forcedState == 0)){
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

    // The search loop above can exit (nPlayout reached, root->forcedState flips nonzero, or a
    // searchStuck early-return in playout()) while an evaluation request is still sitting in the
    // shared evaluator queue, pointing at one of our nodes. The caller (start_self_play) calls
    // jump()/reset() right after this returns, which deletes nodes — so any such request must be
    // waited on and drained here first, or the evaluator thread will read freed memory.
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

    // if(globalConfig.detailedStat)
    //     printVariation();
}

Move MCTS::getMove(float temp){
    for(int i=0; i<10; ++i){
        runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout / 10, globalConfig.time / 10);
        // printVariation();
        const auto& [winProb, scoreEXP] = getEval();
        // std::cout << "winprob : " << winProb << "\nscoreEXP : " << scoreEXP << std::endl;
    }
    return root->selectMove(temp);
}

MoveData MCTS::getMoveProb(float temp){
    runSimulation((globalConfig.mode == "playout") ? PLAYOUT : TIMEOUT, globalConfig.nPlayout, globalConfig.time);
    return root->selectMoveProb(temp);
}

std::pair<float, float> MCTS::getEval(){
    assert(root->N > 0);
    if(root->forcedState == 0)
        return {static_cast<float>(-root->W) / root->N, static_cast<float>(-root->S) / root->N};
    else if(root->forcedState > 0)
        return {1.0f, 0.0f};
    else
        return {-1.0f, 0.0f};
}

void MCTS::printVariation(){
    Node* node = root;

    while(node->N > 1){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
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

            m = node->game.getAvailableMoves()[maxi];
        }
        else if(node->forcedState > 0){
            for(int i=0; i<node->child.size(); ++i){
                if(node->child[i]->forcedState < 0){
                    m = node->game.getAvailableMoves()[i];
                }
            }
            // std::cerr << "only move : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;

            if(node->forcedState == 2)
                break;

            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if(node->game.getAvailableMoves()[i] == m){
                    maxi = i;
                    break;
                }
            }
            // if(maxi == -1){
            //     for(const auto& options : node->game.getAvailableMoves())
            //         std::cerr << "options : " << static_cast<int>(options.first) << " " << static_cast<int>(options.second) << std::endl;
            // }
        }
        else{ // losing. Delay losing as much as possible.
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }

            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }

        int visit = node->edgeN[maxi];
        node = node->child[maxi];
        if(node == nullptr)
            return;
        std::cout << (int)m / colSize << " " << (int)m % colSize << " " << visit << " forced " << node->forcedState << " Q: " << 
            node->W/node->N << " initQ : " << node->initQ << " Wp : " << node->Wp/node->N 
            << " S : " << node->S / node->N << std::endl;  
    }
}

std::vector<Move> MCTS::followUpFrom(Node* node, int maxDepth){
    std::vector<Move> seq;

    while(node != nullptr && node->N > 1 && (int)seq.size() < maxDepth){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if(node->edgeN[i] > maxv){
                    maxv = node->edgeN[i];
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }
        else if(node->forcedState > 0){
            // A node's forcedState can be set at birth (Node::addChild, when the move that
            // created it already ended the game) without ever going through expand() filling
            // in `child` -- expand() deliberately skips that for any already-forced node, so
            // `child` can be shorter than game.getAvailableMoves() (even empty) here. Bound by
            // the smaller of the two so we never index past either array; an empty/partial
            // child list just means no continuation to show, which the maxi==-1 check below
            // already handles gracefully.
            int limit = (int)node->child.size() < (int)node->game.getAvailableMoves().size()
                ? (int)node->child.size() : (int)node->game.getAvailableMoves().size();
            for(int i=0; i<limit; ++i){
                if(node->child[i] != nullptr && node->child[i]->forcedState < 0){
                    m = node->game.getAvailableMoves()[i];
                    maxi = i;
                    break;
                }
            }
            if(maxi == -1 || node->forcedState == 2)
                break;
        }
        else{ // losing. Delay losing as much as possible.
            int limit = (int)node->child.size() < (int)node->game.getAvailableMoves().size()
                ? (int)node->child.size() : (int)node->game.getAvailableMoves().size();
            for(int i=0; i<limit; ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }

        seq.push_back({static_cast<uint8_t>(m / colSize), static_cast<uint8_t>(m % colSize)});
        node = node->child[maxi];
    }

    return seq;
}

void MCTS::printAnalysis(){
    std::cout << "analysis begin" << std::endl;

    if(root->forcedState > 0)
        std::cout << "winrate : 1" << std::endl;
    else if(root->forcedState < 0)
        std::cout << "winrate : -1" << std::endl;
    else if(root->N > 0)
        // Every playout adds root's OWN Wp contribution with a sign already oriented so that
        // root->Wp/root->N == -(visit-weighted average of the *children's* raw Wp/N below) --
        // verified numerically against real search output. Since the per-move winrate below is
        // deliberately left unnegated (child->Wp/N, matching how selectChildInSearch() reads
        // child->W directly as its PUCT preference -- the heaviest-visited child empirically has
        // the *highest* unnegated value, not the lowest), root's own line needs the negation to
        // land on the same scale: "root winrate" ends up the weighted average of the move winrates
        // printed below, rather than their complement.
        std::cout << "winrate : " << (-root->Wp / root->N) << std::endl;
    else
        std::cout << "winrate : 0" << std::endl;
    std::cout << "visits : " << root->N << std::endl;
    // Once forcedState != 0 the engine has proven a win/loss and the "analyze" command's search
    // loop (see ModelCompare::analyze) stops for good right here, well short of the requested
    // playout target -- expose this explicitly so the GUI can tell "search is genuinely done"
    // apart from "still climbing toward the target", instead of just comparing visits to target.
    std::cout << "forced : " << root->forcedState << std::endl;
    // root->initQ is set exactly once, the first time this exact position is ever evaluated by
    // the network (calculateQ's blended-utility output) -- i.e. the raw value-head verdict before
    // any tree search refined it. Comparing it against the searched winrate above shows how much
    // search moved the evaluation away from the network's first guess.
    std::cout << "initQ : " << root->initQ << std::endl;

    // scoreMap / captureMap are per-point NN outputs for the *current* root position, not
    // per-child search stats -- fetch them directly (evaluator->evaluate hits the cache,
    // since root was already evaluated by the very first playout, so this is cheap).
    // scoreMap: tanh output in [-1, 1], matching the training label convention in
    // Game::makeMove's end-of-game scoreMap (-1 = Black-owned point, +1 = White-owned point).
    // captureMap: sigmoid output in [0, 1], probability that the stone at that point gets
    // captured; already masked to 0 at empty points (see PolicyValueNet::batchEvaluate).
    {
        auto buf = std::make_shared<NNResultBuf>();
        evaluator->evaluate(buf, &root->game, root->hashValue);
        const auto& [policy, wp, sd, scoreMap] = *buf->result;
        std::cout << "scoreMap :";
        for(float v : scoreMap) std::cout << " " << v;
        std::cout << std::endl;
        // std::cout << "captureMap :";
        // for(float v : captureMap) std::cout << " " << v;
        std::cout << std::endl;
    }

    // per-move breakdown is only meaningful once the root has been expanded with real moves.
    if(root->forcedState == 0){
        const auto& moves = root->game.getAvailableMoves();
        for(int i=0; i<moves.size(); ++i){
            int r = moves[i] / colSize;
            int c = moves[i] % colSize;

            float visits = (i < root->edgeN.size()) ? root->edgeN[i] : 0.0f;
            float prior = (i < root->edgeP.size()) ? root->edgeP[i] : 0.0f;
            float winrate = 0.0f;
            float q = 0.0f;
            if(i < root->child.size() && root->child[i] != nullptr && root->child[i]->N > 0){
                winrate = root->child[i]->Wp / root->child[i]->N;
                q = root->child[i]->W / root->child[i]->N; // blended utility actually used for PUCT selection
            }

            std::cout << "move " << r << " " << c
                       << " visits " << visits
                       << " prior " << prior
                       << " winrate " << winrate
                       << " q " << q
                       << " variation";
            if(i < root->child.size() && root->child[i] != nullptr){
                for(const Move& fm : followUpFrom(root->child[i], 6)){
                    std::cout << " " << (int)fm.first << " " << (int)fm.second;
                }
            }
            std::cout << std::endl;
        }
    }
    // Once root is forced, every other child's search stats are stale (search stopped the
    // instant forcedState flipped), so the full breakdown above doesn't apply -- but the GUI
    // still wants to see *which* move wins/delays and how the line continues, instead of an
    // empty move list. Find that single move the same way followUpFrom finds it one ply down
    // (a child already proven losing for the opponent when winning, or the least-bad delaying
    // child when losing), then print it as the lone "move" line with its own variation tail.
    else{
        const auto& moves = root->game.getAvailableMoves();
        int limit = (int)root->child.size() < (int)moves.size() ? (int)root->child.size() : (int)moves.size();
        int maxi = -1;
        int maxv = -1;
        if(root->forcedState > 0){
            for(int i=0; i<limit; ++i){
                if(root->child[i] != nullptr && root->child[i]->forcedState < 0){
                    maxi = i;
                    break;
                }
            }
        }
        else{
            for(int i=0; i<limit; ++i){
                if(root->child[i] != nullptr && root->child[i]->forcedState > maxv){
                    maxv = root->child[i]->forcedState;
                    maxi = i;
                }
            }
        }

        if(maxi != -1){
            int r = moves[maxi] / colSize;
            int c = moves[maxi] % colSize;
            Node* chosen = root->child[maxi];

            float visits = (maxi < (int)root->edgeN.size()) ? root->edgeN[maxi] : 0.0f;
            float prior = (maxi < (int)root->edgeP.size()) ? root->edgeP[maxi] : 0.0f;
            float winrate = (chosen->N > 0) ? chosen->Wp / chosen->N : 0.0f;
            float q = (chosen->N > 0) ? chosen->W / chosen->N : 0.0f;

            std::cout << "move " << r << " " << c
                       << " visits " << visits
                       << " prior " << prior
                       << " winrate " << winrate
                       << " q " << q
                       << " variation";
            for(const Move& fm : followUpFrom(chosen, 6)){
                std::cout << " " << (int)fm.first << " " << (int)fm.second;
            }
            std::cout << std::endl;
        }
    }

    std::cout << "analysis end" << std::endl;
}

bool MCTS::jump(Move move){
    Node* old_root = root;
    root = root->jump(move);
    old_root->deleteTree(root);
    return root != nullptr;
}

void MCTS::reset(){
    root->deleteTree();
    root = new Node(Game(), hash.baseHash(), transposTable);

    if(globalConfig.transTable){
        transposTable->clear();
        transposTable->emplace(hash.baseHash(), std::make_pair(root, 1));
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

            // on first visit, set up policyMask then just quit.
            if(cur->N == 0.0f){
                cur->game.setPolicyMask();
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
                if(forced == 0 && cur->evaluation != nullptr){
                    cur->edgeP = softmax(std::get<0>(*(cur->evaluation)), cur);
                    cur->edgeN.assign(cur->edgeP.size(), 0.0f);
                    // no longer needs to store evaluation. Evaluation may be freed.
                    // std::cout << "reset evaluation : " << cur << " " << cur->evaluation << std::endl;
                    cur->evaluation.reset();
                }
            }

            forced = cur->forcedState;
            if(forced != 0)
                break;

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

            if(globalConfig.detailedStat){
                if(forced > 0)
                    propagate(path, -1.0f, -1.0f, -cur->game.scoreDiff(cur->turn));
                else
                    propagate(path, 1.0f, 1.0f, -cur->game.scoreDiff(cur->turn));
            }
            else{
                if(forced > 0)
                    propagate(path, -1.0f);
                else
                    propagate(path, 1.0f);
            }
        }
        else{ // if final search node is non-determined node, ask for evaluation
            // enqueue evaluation
            auto buf = std::make_shared<NNResultBuf>();

            // TODO : move game.getAvailableMoves(), transferTable into game object. They are used both in NN evaluation & expansion.
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
    if(inEvaluation.size() >= globalConfig.search_thread_num || (playMode == PLAYOUT && searchCounter == nPlayout && !inEvaluation.empty()) ||
    (root->forcedState != 0 && !inEvaluation.empty()) || searchStuck){
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

    // instead updating edgeP and edgeN right away, store the evaluation and only update edgeP and edgeN after expansion.
    cur->evaluation = buf->result;
    // std::cout << "get evaluation : " << cur << " " << buf->result << " " << cur->evaluation << std::endl;

    if(globalConfig.detailedStat){ // if detailedStat = true, then update S, Wp variable. Else, ignore those values.
        std::tie(cur->initQ, evalW) = calculateQ(buf->result, cur->game);
        evalS = std::get<2>(*(buf->result));
        evalQ = cur->initQ;
    }
    else{
        evalQ = calculateQ(buf->result, cur->game).first;
    }

    propagate(path, evalQ, evalW, evalS);
}


void MCTS::propagate(const std::vector<Node*>& path, float evalQ, float evalW, float evalS){
    if(globalConfig.detailedStat){ // if detailedStat = true, update S, Wp variable as well. Otherwise, ignore those.
        for (Node* n : path | std::views::reverse) {
            n->W += 1.0f;   // revert VL
            n->W += evalQ;
            n->Wp += evalW;
            n->S += evalS;
            evalQ = -evalQ;
            evalS = -evalS;
            evalW = -evalW;
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
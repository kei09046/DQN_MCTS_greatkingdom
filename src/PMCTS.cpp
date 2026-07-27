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
        const auto& availableMoves = node->availableMoves_();
        const auto& transferTable = node->TransferTable_();

        const int moveSize = availableMoves.size();
        assert(transferTable.size() == moveSize);
        std::vector<float> n_logit(moveSize);

        std::vector<int> maxIdx(moveSize);
        for(int i=0; i<moveSize; ++i){
            n_logit[i] = logit[availableMoves[i].first * colSize + availableMoves[i].second];
            maxIdx[i] = availableMoves[i].first * colSize + availableMoves[i].second;
            for(int j : transferTable[i]){
                if(logit.at(j) > n_logit[i]){
                    n_logit[i] = logit[j];
                    maxIdx[i] = j;
                }
            }
        }

        std::vector<float> exp_logit(moveSize);
        float max_logit = *std::max_element(n_logit.begin(), n_logit.end()); // For numerical stability

        // Compute exponentials after subtracting max_logit
        float sum_exp = 0.0f;
        for (int i = 0; i < moveSize; ++i) {
            exp_logit[i] = std::exp(n_logit[i] - max_logit);
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
        //                 if(availableMoves[t] == Move{i, j}){
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
        //                 if(availableMoves[t] == Move{i, j}){
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
        const auto& [logAct, winP, scoreEXP, scoreMap, captureMap] = *nnOutput;

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
                    avgCapChance += captureMap[cur];
                    cur = game.getStone({cur/colSize, cur%colSize}).next;
                } while (cur != head);

                avgCapChance /= c.size;

                int type = game.getBoard({i / colSize, i % colSize}) == turn ? 0 : 1;
                captureV[type] = (captureV[type] > avgCapChance) ? captureV[type] : avgCapChance;
            }
        }

        // float utility = winP * 0.9f;
        float utility = winP * 0.9f + (captureV[0] - captureV[1]) * 0.03f + scoreV * 0.01f;
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
N(0.0f), W(0.0f), initQ(0.0f), S(0.0f), Wp(0.0f), forcedState(0), onlyMove(RESIGNMOVE), hashValue(hashValue), expanded(false), evaluation(nullptr), transposTable(transposTable){
}

void Node::addChild(const Move& move, int idx){
    HashValue newHash = hash.computeHashAfterMove(game, move, hashValue);

    Node* childNode;
    if(globalConfig.transTable){
        auto it = transposTable->find(newHash);

        if(it == transposTable->end()){
            Game ng = this->game;
            // if idx = -1, move is not in the list. -> Add completely new child.
            if(idx == -1){
                ng.makeMove(move);
            }
            // else, compute child based on stats calculated on expand() call.
            else{
                ng.makeMoveGivenScore(move, transferTable[idx]);
            }
            childNode = new Node(ng, newHash, transposTable);
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
            ng.makeMove(move);
        }
        else{
            ng.makeMoveGivenScore(move, transferTable[idx]);
        }
        childNode = new Node(ng, newHash, transposTable);
    }

    if(idx == -1)
        child.push_back(childNode);
    else
        child[idx] = childNode;
}

void Node::threatCheck(){
    auto [threat, forced] = game.threatCheck();
    assert(threat != RESIGNMOVE || forced == 0);
    forcedState = forced;
    onlyMove = threat;
}

void Node::expand(){
    //ModelCompare::displayBoardGUI(true, game);
    //std::cerr << "Expand called" << std::endl;

    expanded = true;
    // if forcedState != 0, then there is either a winning move or there will be no available moves anyway. Don't expand.
    if(forcedState != 0)
        return;

    auto [result, moves, tranfTable] = game.expand(onlyMove);

    transferTable = std::move(tranfTable);
    onlyMove = std::move(result.first);
    forcedState = std::move(result.second);

    child = std::vector<Node*>(moves.size(), nullptr);
    availableMoves = std::move(moves);

    if(availableMoves.empty()){
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

    if(availableMoves.empty()){
        std::cerr << N << " " << W << " " << initQ << " " << S << " " << Wp << std::endl;
        ModelCompare::displayBoardGUI(true, game);
    }
    assert(!availableMoves.empty());

    for(int i=0; i<availableMoves.size(); ++i){
        if(child[i] == nullptr){
            pref = ((globalConfig.fpu < 0.0f) ? 0.0f : -W/N-globalConfig.fpu) + globalConfig.cPuct * edgeP[i] * sqrt(N);
            lost = false;
        }

        else{
            int forced = child[i]->forcedState;
            
            // if winning continuation found.
            if(forced < 0){
                forcedState = -forced + 1;
                onlyMove = availableMoves[i];
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
        addChild(availableMoves[maxi], maxi);
    return maxi;
}

Move Node::selectMove(float temp){
    //std::cout << "available move size : " << availableMoves.size() << std::endl;

    if(forcedState > 0){
        // if(globalConfig.detailedStat)
        //     std::cout << "status: " << static_cast<int>(onlyMove.first) << " " << static_cast<int>(onlyMove.second)
        //     << " forced : " << -forcedState + (forcedState > 0 ? 1 : -1) << std::endl;
        return onlyMove;
    }
    else if(availableMoves.empty()){
        return RESIGNMOVE;
    }
    // else if(globalConfig.detailedStat){
    //     std::vector<int> v(availableMoves.size());
    //     std::iota(v.begin(), v.end(), 0);
    //     std::sort(v.begin(), v.end(), [&](const int& a, const int& b){
    //         return edgeN[a] > edgeN[b];
    //     });

    //     for(int i=0; i<std::min(static_cast<int>(availableMoves.size()), 3); ++i){
    //         int idx = v[i];
    //         if(child[idx] != nullptr)
    //             std::cout << "status: " << static_cast<int>(availableMoves[idx].first) << " " << static_cast<int>(availableMoves[idx].second)
    //             << " forced : " << child[idx]->forcedState << " sc: " << edgeN[idx] << " Q: " 
    //             << child[idx]->W/child[idx]->N << " initQ : " << child[idx]->initQ << " Wp : " << child[idx]->Wp/child[idx]->N 
    //             << " S : " << child[idx]->S / child[idx]->N << " P " << edgeP[idx] << std::endl;
    //     }
    // }

    int maxi, maxn = -1, index;
    for(int i=0; i<availableMoves.size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
    }

    if(temp >= 5.0f || game.getMoveCount() >= 10){
        return availableMoves[maxi];
    }

    std::vector<float> weights(availableMoves.size());
    std::vector<float> cumulative(availableMoves.size());
    for(int i=0; i<availableMoves.size(); ++i){
        weights[i] = std::pow(edgeN[i], temp);
    }
    std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

    std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
    float rnd = dist(gen);

    auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
    index = std::distance(cumulative.begin(), it);
    return availableMoves[index];

}


MoveData Node::selectMoveProb(float temp){
    std::vector<float> visitPortion(outputSize, 0.0f);
    Move selectedMove;

    if(forcedState == -1){
        assert(availableMoves.empty());
        return {RESIGNMOVE, visitPortion};
    }

    if(forcedState > 0){
        // label smoothing is applied here.
        //std::fill(visitPortion.begin(), visitPortion.end(), 0.02f/(outputSize - 1));
        visitPortion[onlyMove.first * colSize + onlyMove.second] = 1.0f;
        selectedMove = onlyMove;
    }

    else if(temp >= 5.0f || game.getMoveCount() >= 10){
        int maxi, maxn = -1;
        for(int i=0; i<availableMoves.size(); ++i){
            if(edgeN[i] > maxn){
                maxn = edgeN[i];
                maxi = i;
            }
            visitPortion[availableMoves[i].first * colSize + availableMoves[i].second] = edgeN[i]/N;
        }
        selectedMove = availableMoves[maxi];
    }

    else{
        std::vector<float> cumulative(availableMoves.size()), weights(availableMoves.size());

        for(int i=0; i<availableMoves.size(); ++i){
            visitPortion[availableMoves[i].first * colSize + availableMoves[i].second] = edgeN[i]/N;
            weights[i] = (edgeN[i] == 0.0f) ? 0.0f : std::pow(edgeN[i], temp);
        }

        std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

        std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
        float rnd = dist(gen);

        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
        int index = std::distance(cumulative.begin(), it);
        selectedMove = availableMoves[index];
    }

    return {selectedMove, visitPortion};
}

Node* Node::jump(Move move){
    if(N == 0){
        // std::cerr << "original forcedState : " << forcedState << std::endl;
        // std::cerr << "threat : ";
        // printMove(onlyMove);
        threatCheck();
        // std::cerr << "after forcedState : " << forcedState << std::endl;
        // std::cerr << "threat : ";
        // printMove(onlyMove);
    }
    if(!expanded){
        // std::cerr << "E original forcedState : " << forcedState << std::endl;
        // std::cerr << "threat : ";
        // printMove(onlyMove);
        expand();
        // std::cerr << "E after forcedState : " << forcedState << std::endl;
        // std::cerr << "threat : ";
        // printMove(onlyMove);
    }
    N++;

    // std::cerr << "requested move : " << static_cast<int>(move.first) << "," << static_cast<int>(move.second) << std::endl;
    // std::cerr << "available options : " << std::endl;
    // for(auto p : availableMoves)
    //     std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";
    // std::cerr << "node's state : " << std::endl;
    // ModelCompare::displayBoardGUI(true, game);

    for(int i=0; i<availableMoves.size(); ++i){
        if(availableMoves[i] == move){
            // delay child allocation as much as possible. Happens when agent is forced to jump to a move that it hasn't considered at all.
            if(child[i] == nullptr){
                addChild(move, i);
            }
            return child[i];
        }
    }

    // if no child matches the move, add one. Only happens when human opponent makes suboptimal move.
    // std::cerr << "unexpected move!" << std::endl;
    // addChild(move.first, move.second, nGame);
    // return child.back();

    std::cerr << "warning! jump to illegal location!" << std::endl;
    std::cerr << "requested move : " << static_cast<int>(move.first) << "," << static_cast<int>(move.second) << std::endl;
    std::cerr << "expanded : " << expanded << std::endl;
    std::cerr << "N : " << N << std::endl;
    std::cerr << "forcedState : " << forcedState << std::endl;
    std::cerr << "only move : " << static_cast<int>(onlyMove.first) << " " << static_cast<int>(onlyMove.second) << std::endl;
    std::cerr << "available options : " << std::endl;
    for(auto p : availableMoves)
        std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";

    std::cerr << "node's state : " << std::endl;
    ModelCompare::displayBoardGUI(true, game);
    
    std::cerr << "transfer table : " << std::endl;
    for(const auto& v : transferTable){
        for(const auto& move : v){
            std::cerr << static_cast<int>(move) / colSize << static_cast<int>(move) % colSize << " ";
        }
        std::cerr << std::endl;
    }

    return nullptr;
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
        threatCheck();
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
    if(forcedState <= 0 && availableMoves.size() > 0){
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
        Move m;

        if(node->forcedState == 0){
            for(int i=0; i<node->availableMoves.size(); ++i){
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

            m = node->availableMoves[maxi];
        }
        else if(node->forcedState > 0){
            m = node->onlyMove;
            assert(m != RESIGNMOVE);
            // std::cerr << "only move : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;

            if(node->forcedState == 2)
                break;

            for(int i=0; i<node->availableMoves.size(); ++i){
                if(node->availableMoves[i] == m){
                    maxi = i;
                    break;
                }
            }
            // if(maxi == -1){
            //     for(const auto& options : node->availableMoves)
            //         std::cerr << "options : " << static_cast<int>(options.first) << " " << static_cast<int>(options.second) << std::endl;
            // }
        }
        else{ // losing. Delay losing as much as possible.
            for(int i=0; i<node->availableMoves.size(); ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }

            if(maxi == -1)
                break;
            m = node->availableMoves[maxi];
        }

        int visit = node->edgeN[maxi];
        node = node->child[maxi];
        if(node == nullptr)
            return;
        std::cout << (int)m.first << " " << (int)m.second << " " << visit << " forced " << node->forcedState << " Q: " << 
            node->W/node->N << " initQ : " << node->initQ << " Wp : " << node->Wp/node->N 
            << " S : " << node->S / node->N << std::endl;  
    }
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

            // One can save memory and CPU expansion by delaying expansion until second visit. However, that would also
            // mean forcedState would not be computed in first visit thus NN evaluation request would happen for terminal nodes.
            // TODO : Is it possible to separate expand() logic to legal move computation + actual expansion?
            // Also, is it possible to expand each child separately while maintaining speed advantage?

            // step 1 done. Expansion is only done on second visit.
            
            // step 2 done. Split expand logic so that terminal state is determined as much as possible on first visit without actual expansion.

            // TODO step 3. When expand called, only expand the nodes which gives more score. Other nodes should be expanded only when they are actually visited.

            // on first visit, check easy capture then just quit.
            if(cur->N == 0.0f){
                cur->threatCheck();
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
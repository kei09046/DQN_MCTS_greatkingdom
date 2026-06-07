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
        //std::cerr << "softmax start" << std::endl;
        const auto& availableMoves = node->availableMoves_();
        const auto& transferTable = node->TransferTable_();

        const int moveSize = availableMoves.size();
        std::vector<float> n_logit(moveSize);

        for(int i=0; i<moveSize; ++i){
            n_logit[i] = logit.at(availableMoves[i].first * colSize + availableMoves[i].second);
        }

        for(int i=0; i<transferTable.size(); ++i){
            const auto idx = transferTable[i][0];
            for(int j=1; j<transferTable[i].size(); ++j){
                n_logit.at(idx) = std::max(n_logit.at(idx), logit[j]);
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
        //std::cerr << "softmax done" << std::endl;
        return exp_logit;
    }

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

        // float capChance = std::max(captureV[0], captureV[1]);
        // float capChanceClip = std::min(std::max(capChance, 0.25f), 0.75f);
        float utility = winP * 0.9f + (captureV[0] - captureV[1]) * 0.03f + scoreV * 0.03f;
        if(globalConfig.detailedStat){
            static int cntr = 0;
            if(cntr++ % 100 == 0){
                ModelCompare::displayBoardGUI(true, game);
                std::cout << winP << " " << captureV[0] - captureV[1] << " " << scoreV << " " << utility << std::endl;
                for(int i=0; i<rowSize; ++i){
                    for(int j=0; j<colSize; ++j){
                        std::cout << scoreMap[i * colSize + j] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                for(int i=0; i<rowSize; ++i){
                    for(int j=0; j<colSize; ++j){
                        std::cout << captureMap[i * colSize + j] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }

        return {utility, winP};
    }
}

// N : # of visits, W : total action-value Q : mean action-value P : prior policy evaluation; stored by parent
Node::Node(const Game& g, const HashValue hashValue, TransTable* const transposTable):
game(g), turn(g.getTurn()), 
N(0.0f), W(0.0f), initQ(0.0f), S(0.0f), Wp(0.0f), forcedState(0), winmove(RESIGNMOVE), hashValue(hashValue), transposTable(transposTable){
    // if(hashValue == (HashValue)12122450572009219436){
    //     std::cerr << "12122450572009219436 : " << std::endl;
    //     ModelCompare::displayBoardGUI(false, g);
    // }
}

void Node::addChild(const Move& move, const Game& ng){
    HashValue newHash = hash.computeHashAfterMove(game, move, hashValue);
    Node* childNode;

    if(globalConfig.transTable){
        auto it = transposTable->find(newHash);

        if(it == transposTable->end()){
            childNode = new Node(ng, newHash, transposTable);
            transposTable->emplace(newHash, std::make_pair(childNode, 1));
        }
        else{
            //std::cerr << "duplicate hashValue : " << newHash << " move : " << static_cast<int>(move.first) << " " << static_cast<int>(move.second) << std::endl;
            //ModelCompare::displayBoardGUI(false, ng); 
            childNode = it->second.first;
            (it->second.second)++;
        }
    }
    else{
        childNode = new Node(ng, newHash, transposTable);
    }

    child.push_back(childNode);
    //std::cerr << "adding! " << r << " " << c << std::endl;
    //std::cerr << "adding done!" << std::endl;
}

void Node::expand(){
    //ModelCompare::displayBoardGUI(true, game);
    //std::cerr << "Expand called" << std::endl;
    auto [result, moves, games, tranfTable] = game.expand();

    // printMove(result.first);
    // for(int i=0; i<moves.size(); ++i){
    //     printMove(moves[i]);
    //     ModelCompare::displayBoardGUI(true, games[i]);
    // }

    // for(const auto& v : tranfTable){
    //     for(const auto idx : v){
    //         std::cerr << idx << " ";
    //     }
    //     std::cerr << std::endl;
    // }

    transferTable = std::move(tranfTable);
    winmove = std::move(result.first);
    forcedState = std::move(result.second);

    for (int i=0; i<games.size(); ++i) {
        addChild(moves[i], std::move(games[i])); 
    }
    availableMoves = std::move(moves);

    if(availableMoves.empty()){
        // std::cerr << "there are no legal moves! turn : " << static_cast<int>(game.getTurn()) << " " << static_cast<int>(turn) << std::endl;
        // ModelCompare::displayBoardGUI(true, game);
        forcedState = -1;
    }
    
    // // //std::cerr << "expanding!" << std::endl;
    // std::bitset<outputSize> candidateLegal; // mark candidate legal moves

    // // improve capture check performance by checking if there is any group with liberty count 1.
    // Move threat = RESIGNMOVE;

    // for(int i=0; i<rowSize; ++i){
    //     for(int j=0; j<colSize; ++j){
    //         const Chain c = game.getChain({i, j});
    //         if(c.size != 0 && c.liberties.count() == 1){
    //             auto color = game.getBoard({i, j});
    //             int onlyLib = c.liberties._Find_first();

    //             if(game.isLegal(onlyLib / colSize, onlyLib % colSize)){
    //                 // if my stone is under threat -> have to find only move unless can capture opponent's stone.
    //                 if(color == game.getTurn()){
    //                     threat = {onlyLib / colSize, onlyLib % colSize};
    //                 }

    //                 // if opponent stone is capturable
    //                 else{
    //                     winmove = {onlyLib / colSize, onlyLib % colSize};
    //                     forcedState = 2;
    //                     return;
    //                 }
    //             }
    //         }
    //     }
    // }
    // candidateLegal = game.getLegalMoves();
    // // can only pass if it's beneficial
    // candidateLegal[outputSize - 1] = (game.scoreWinner() == game.getTurn());

    // std::vector<Game> nextGames(boardSize + 1); // +1 for pass
    // // update scores & remove useless moves
    // for(int idx = 0; idx < boardSize + 1; ++idx){
    //     if(candidateLegal[idx]){
    //         uint8_t r = idx / colSize;
    //         uint8_t c = idx % colSize;
    //         nextGames[idx] = game;
    //         auto [clr, wintype] = nextGames[idx].makeMove({r, c});

    //         if(clr == turn){ // there is immediate win by score. win in 1.
    //             forcedState = 2;
    //             winmove = {r, c};
    //             return;
    //         }

    //         // there is immediate capture next move, or the move is self-suicidal.
    //         else if((threat != RESIGNMOVE && (nextGames[idx].isLegal(threat) || wintype == CAPTURE))){
    //             candidateLegal[idx] = false;
    //         }

    //         else{
    //             candidateLegal &= nextGames[idx].getLegalMoves();
    //             candidateLegal[idx] = true; // keep itself
    //         }
    //     }
    // }

    // if(candidateLegal.none()){ // if there are no moves, mark it as loss.
    //     forcedState = -1;
    //     return;
    // }
    
    // // finally add child
    // int cntr = 0;
    // for(uint8_t idx = 0; idx < outputSize; ++idx){
    //     if(candidateLegal[idx]){
    //         availableMoves.push_back({idx / colSize, idx % colSize});
    //         addChild({idx/colSize, idx%colSize}, nextGames[idx]);
    //         const auto acquired = (game.getLegalMoves() ^ nextGames[idx].getLegalMoves()).set(boardSize, false);
    //         // if any points of territory is acquired
    //         if(acquired.count() > 1){
    //             std::vector<uint8_t> acquiredV;
    //             acquiredV.reserve(acquired.count() + 1);
    //             // acquiredV : {which move it indicates, terr 1, terr 2, ...}
    //             acquiredV.push_back(cntr++);
    //             for (size_t i = acquired._Find_first(); i < boardSize; i = acquired._Find_next(i)) {
    //                 acquiredV.push_back(i);
    //             }
    //             transferTable.push_back(std::move(acquiredV));
    //         }
    //     }
    // }


    // assert(threat == RESIGNMOVE || candidateLegal.count() == 1);
    //std::cerr << "expand finished" << std::endl;
    #ifdef measureTime
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    expandTime += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    #endif
}

int Node::selectChildInSearch(){
    int maxi = -1;
    float pref, maxval = -1000.0f; // pref may be less than -1.(due to score head)
    bool lost = true;

    assert(!availableMoves.empty());
    for(int i=0; i<availableMoves.size(); ++i){
        int forced = child[i]->forcedState;
        
        // if winning continuation found.
        if(forced < 0){
            forcedState = -forced + 1;
            winmove = availableMoves[i];
            return i;
        }
        // only select non-losing move.
        if(forced == 0){
            pref = ((edgeN[i] == 0.0f) ? ((globalConfig.fpu < 0.0f) ? 0.0f : -W/N-globalConfig.fpu) : child[i]->W / child[i]->N) 
            + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
            lost = false;
        }
        // losing move can be selected. Just mark as loss.
        else{ 
            pref = -1.0f + globalConfig.cPuct * edgeP[i] * sqrt(N)/(1 + edgeN[i]);
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
    return maxi;
}

Move Node::selectMove(float temp){
    //std::cout << "available move size : " << availableMoves.size() << std::endl;

    if(forcedState > 0){
        if(globalConfig.detailedStat)
            std::cout << "status: " << static_cast<int>(winmove.first) << " " << static_cast<int>(winmove.second)
            << " forced : " << -forcedState + (forcedState > 0 ? 1 : -1) << std::endl;
        return winmove;
    }
    else if(availableMoves.empty()){
        return RESIGNMOVE;
    }
    else if(globalConfig.detailedStat){
        std::vector<int> v(availableMoves.size());
        std::iota(v.begin(), v.end(), 0);
        std::sort(v.begin(), v.end(), [&](const int& a, const int& b){
            return child[a]->N > child[b]->N;
        });

        for(int i=0; i<std::min(static_cast<int>(availableMoves.size()), 3); ++i){
            int idx = v[i];
            std::cout << "status: " << static_cast<int>(availableMoves[idx].first) << " " << static_cast<int>(availableMoves[idx].second)
            << " forced : " << child[idx]->forcedState << " sc: " << edgeN[idx] << " Q: " 
            << child[idx]->W/child[idx]->N << " initQ : " << child[idx]->initQ << " Wp : " << child[idx]->Wp/child[idx]->N 
            << " S : " << child[idx]->S / child[idx]->N << " P " << edgeP[idx] << std::endl;
        }
    }

    std::vector<float> weights(availableMoves.size());
    std::vector<float> cumulative(availableMoves.size());

    int maxi, maxn = -1, index;
    for(int i=0; i<availableMoves.size(); ++i){
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
        return availableMoves[index];
    }

    return availableMoves[maxi];
}


MoveData Node::selectMoveProb(float temp){
    std::vector<float> visitPortion(outputSize, 0.0f);

    if(forcedState > 0){
        visitPortion[winmove.first * colSize + winmove.second] = 1.0f;
        return {winmove, visitPortion};
    }
    if(availableMoves.empty()){
        return {RESIGNMOVE, visitPortion};
    }
    std::vector<float> cumulative(availableMoves.size()), weights(availableMoves.size());
    int maxi, maxn = -1;
    for(int i=0; i<availableMoves.size(); ++i){
        if(edgeN[i] > maxn){
            maxn = edgeN[i];
            maxi = i;
        }
        weights[i] = std::pow(edgeN[i], temp);
        visitPortion[availableMoves[i].first * colSize + availableMoves[i].second] = edgeN[i]/N;
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
        //printMove(availableMoves[index]);
        return {availableMoves[index], visitPortion};
    }

    return {availableMoves[maxi], visitPortion};
}

Node* Node::jump(Move move){
    if(N == 0){
        expand();
        N++;
    }

    int idx = -1;
    for(int i=0; i<availableMoves.size(); ++i){
        if(availableMoves[i] == move){
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
    for(auto p : availableMoves)
        std::cerr << static_cast<int>(p.first) << "," << static_cast<int>(p.second) << " ";

    std::cerr << "node's state : " << std::endl;
    ModelCompare::displayBoardGUI(false, game);
    
    std::cerr << "transfer table" << std::endl;
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
            c->deleteTree();
        }
        delete this;
        return;
    }

    auto it = transposTable->find(hashValue);
    if(it == transposTable->end()){
        std::cerr << "hash : " << hashValue << std::endl;
        ModelCompare::displayBoardGUI(false, this->game);
    }
    assert(it != transposTable->end());

    if (--(it->second.second) == 0) {
        for(Node* c : child){
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
            if(c != exception)
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
            if(c != exception)
                c->deleteTree();
        }
        // std::cerr << "deleted : " << hashValue << std::endl;
        transposTable->erase(it);
        delete this;
        return;
    }
}

void Node::addDirichletNoise(Evaluator* evaluator){
    if (N == 0) {
        expand();
        if(forcedState == 0){
            auto buf = std::make_shared<NNResultBuf>();
            evaluator->evaluate(buf, &game, hashValue);

            edgeP = softmax(std::get<0>(*(buf->result)), this);
            edgeN.assign(edgeP.size(), 0.0f);
        }
    }

    if(winmove == RESIGNMOVE && availableMoves.size() > 0){
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
            m = node->winmove;
            assert(m != RESIGNMOVE);
            std::cerr << "winmove : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;

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
                if(node->child[i]->forcedState > maxv){
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

            if (cur->N == 0.0f && (cur != root || !globalConfig.dirichletNoise)) { // first time visit
                cur->expand();
                forced = cur->forcedState;
                break;
            }

            forced = cur->forcedState;
            if(forced != 0)
                break;

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
                cur->edgeP = softmax(evalP, cur);
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
            cur->edgeP = softmax(evalP, cur);
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
//     //     std::cerr << static_cast<int>(path[i]->availableMoves[childIdx[i]].first) << 
//     //     static_cast<int>(path[i]->availableMoves[childIdx[i]].second) << " ";
//     // std::cerr << std::endl;

//     for(int i=childIdx.size() - 1; i >= 0; --i){
//         // on Node n, made move nextMove.
//         n = path[i];

//         if(forced < 0){ // child node is forced loss.
//             forced = -forced + (forced > 0 ? -1 : 1); // loss in 1 -> win in 2.
//             n->forcedState = forced;
//             n->winmove = (n->availableMoves)[childIdx[i]]; // check winning move as only move
//         }
//         else{ // child node is forced win.
//             n->losingMoveCount++;

//             // every option is lost.
//             if(n->losingMoveCount == n->availableMoves.size()){
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
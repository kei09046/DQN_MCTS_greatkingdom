// #include "gamerules.h"
// #include "multimcts.h"
// #include "random.h"
// #include <vector>
// #include <utility>
// #include <cmath>
// #include <iostream>
// #include <random>
// #include <numeric>
// #include <thread>
// #include <atomic>
// #include "MPMCQueue.h"

// #ifdef measureTime
// std::atomic<int> expandTime = 0;
// std::atomic<int> expandWaitTime = 0;
// std::atomic<int> evaluateTime = 0;
// std::atomic<int> searchWaitTime = 0;
// std::atomic<int> searchTime = 0;
// std::atomic<int> makeMoveTime = 0;
// std::atomic<int> copyTime = 0;
// std::atomic<int> extraTime = 0;
// #endif

// Evaluator::Evaluator(PolicyValueNet& net) : net(net), stopWorker(false), evalQueue(searchThreadN) {  // Adjust queue size as needed
//     for(int i=0; i<evaluateThreadN; ++i)
//         worker[i] = std::thread(&Evaluator::evalWorker, this);
// }

// Evaluator::~Evaluator() {
//     stopWorker = true;
//     for (std::thread& work : worker) {
//         work.join();
//     }
// }

// std::future<PolicyValueOutput> 
// Evaluator::enqueueEvaluation(const Game* game, std::vector<std::pair<int, int>>* legal) {
//     std::promise<PolicyValueOutput> evalPromise;
//     std::future<PolicyValueOutput> evalFuture = evalPromise.get_future();

//     evalQueue.emplace(game, legal, std::move(evalPromise));
//     evalCondVar.notify_one();
//     return evalFuture;
// }

// void Evaluator::evalWorker() {
//     while (true) {
//         std::unique_lock<std::mutex> lock(evalMutex);
//         evalCondVar.wait_for(lock, std::chrono::milliseconds(100), [this] {
//             return stopWorker || (evalQueue.size() >= minibatchSize) || (evalQueue.size() >= activeSearchThread.load(std::memory_order_acquire) &&
//         activeSearchThread.load(std::memory_order_acquire) > 0);
//         });

//         if (stopWorker && evalQueue.empty()) {
//             return;
//         }

//         if(evalQueue.empty())
//             continue;
            
//         int size = evalQueue.size();
//         std::vector<const Game*> games;
//         std::vector<std::vector<std::pair<int, int>>*> legalMoves;
//         std::vector<std::promise<PolicyValueOutput>> promises;

//         games.reserve(size);
//         legalMoves.reserve(size);
//         promises.reserve(size);

//         for (int i = 0; i < size; ++i) {
//             EvalRequest req;
//             if (!evalQueue.try_pop(req)) {
//                 break;
//             }

//             games.push_back(req.game);
//             legalMoves.push_back(req.legal);
//             promises.push_back(std::move(req.promise));
//         }

//         lock.unlock(); // Unlock before expensive computation

//         auto results = net.batchEvaluate(games, legalMoves);

//         for (size_t i = 0; i < promises.size(); ++i) {
//             promises[i].set_value({results.first[i], results.second[i]});
//         }
//     }
// }

// std::vector<float> Node::softmax(std::vector<float>& logit){
//     std::vector<float> exp_logit(logit.size());
//     float max_logit = *std::max_element(logit.begin(), logit.end()); // For numerical stability

//     // Compute exponentials after subtracting max_logit
//     float sum_exp = 0.0f;
//     for (size_t i = 0; i < logit.size(); ++i) {
//         exp_logit[i] = std::exp(logit[i] - max_logit);
//         sum_exp += exp_logit[i];
//     }

//     // Normalize
//     for (float& val : exp_logit) {
//         val /= sum_exp;
//     }

//     return exp_logit;
// }

// Node::Node(Game* g): game(g){
//     turn = game->getTurn();
    
//     // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
//     N = 0;
//     W = 0;
//     P = 0.0f;
//     initQ = 0.0f;
//     winmove = {-1, -1};
//     child.reserve(boardSize);
//     legal.reserve(boardSize);
// }

// void Node::expand(){
//     std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

//     Color clr;
//     Game* pass;

//     for(int i=0; i<rowSize; ++i){
//         for(int j=0; j<colSize; ++j){
//             if(game->isLegal(i, j)){
//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point copyBegin = std::chrono::steady_clock::now();
//                 #endif
                
//                 Game* ng = new Game(*game);

//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point copyEnd = std::chrono::steady_clock::now();
//                 copyTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(copyEnd - copyBegin).count());
//                 #endif

//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point moveBegin = std::chrono::steady_clock::now();
//                 #endif

//                 clr = ng->makeMove(i, j);

//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point moveEnd = std::chrono::steady_clock::now();
//                 makeMoveTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(moveEnd - moveBegin).count());
//                 #endif

//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point extraBegin = std::chrono::steady_clock::now();
//                 #endif

//                 if(clr == EMPTY){
//                     child.emplace_back(new Node(ng));
//                     legal.emplace_back(i, j);
//                 }

//                 else if(clr == turn){
//                     delete ng;
//                     winmove = {i, j};

//                     #ifdef measureTime
//                     std::chrono::steady_clock::time_point extraEnd = std::chrono::steady_clock::now();
//                     extraTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - extraBegin).count());
//                     expandTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - begin).count());
//                     #endif
//                     return;
//                 }
//                 std::chrono::steady_clock::time_point extraEnd = std::chrono::steady_clock::now();
//                 #ifdef measureTime
//                 extraTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(extraEnd - extraBegin).count());
//                 #endif
//             }
//         }
//     }

//     pass = new Game(*game);
//     pass->makeMove(rowSize, 0);
//     child.push_back(new Node(pass));
//     legal.emplace_back(rowSize, 0); // pass

//     #ifdef measureTime
//     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//     expandTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//     #endif
// }

// void Node::deletetree(){
//     //std::cout << "delete tree " << std::endl;
//     for(Node* i : child){
//         i->deletetree();
//         delete i;
//     }
// }

// Node::~Node(){
//     delete game;
// }

// float Node::searchandPropagate(Evaluator& evaluator) {
//     W.fetch_add(-virtualLoss);
//     N.fetch_add(1, std::memory_order_relaxed);

//     if (!expanded.load(std::memory_order_acquire)) {
//         #ifdef measureTime
//         std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//         #endif

//         std::lock_guard<std::mutex> lock(expandMutex);

//         #ifdef measureTime
//         std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//         expandWaitTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//         #endif

//         if (!expanded.load(std::memory_order_acquire)) {
//             expand();
//             expanded.store(true, std::memory_order_release);
//         }
//     }

//     if (winmove.first >= 0) {
//         W.fetch_add(-1.0f + virtualLoss);
//         return 1.0f * decay;
//     }
//     if (legal.size() == 1 && game->scoreWinner() != game->getTurn()) {
//         W.fetch_add(1.0f + virtualLoss);
//         return -1.0f * decay;
//     }

//     if (!evaluated.load(std::memory_order_acquire)) {
    
//         if (!evaluating.load(std::memory_order_acquire)) {
//             #ifdef measureTime
//             std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//             #endif

//             std::unique_lock<std::mutex> lock(evaluationMutex);

//             #ifdef measureTime
//             std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//             searchWaitTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//             #endif
            
//             if (!evaluating.load(std::memory_order_acquire)) {  
//                 evaluating.store(true, std::memory_order_release);
//                 lock.unlock(); 
                
//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//                 #endif

//                 auto evalFuture = evaluator.enqueueEvaluation(game, &legal);
//                 auto [logp, q] = evalFuture.get();
//                 auto p = softmax(logp);
    
//                 for (int i = 0; i < legal.size(); ++i) {
//                     child[i]->P = p[i];
//                 }
    
//                 initQ = q;
//                 W.fetch_add(q + virtualLoss);
//                 evaluated.store(true, std::memory_order_release);

//                 #ifdef measureTime
//                 std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//                 evaluateTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//                 #endif

//                 return -q * decay;
//             }
//         }
    
//         #ifdef measureTime
//         std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//         #endif

//         int maxi = -1;
//         float pref, maxval = -1.0f;
    
//         for (int i = 0; i < legal.size(); ++i) {
//             pref = (child[i]->W.load(std::memory_order_relaxed) /
//                     (child[i]->N.load(std::memory_order_relaxed) + 1));
    
//             if (maxval < pref) {
//                 maxval = pref;
//                 maxi = i;
//             }
//         }

//         #ifdef measureTime
//         std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//         searchTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//         #endif
    
//         float r = child[maxi]->searchandPropagate(evaluator);
//         W.fetch_add(r + virtualLoss);
//         return -r * decay;
//     }
    
//     #ifdef measureTime
//     std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//     #endif

//     int maxi = -1;
//     float pref, maxval = -1.0f;

//     for (int i = 0; i < legal.size(); ++i) {
//         pref = (child[i]->W.load(std::memory_order_relaxed) /
//                 (child[i]->N.load(std::memory_order_relaxed) + 1)) +
//                 cPuct * child[i]->P * sqrt(N.load(std::memory_order_relaxed) - 1) /
//                     (1 + child[i]->N.load(std::memory_order_relaxed));

//         if (maxval < pref) {
//             maxval = pref;
//             maxi = i;
//         }
//     }

//     #ifdef measureTime
//     std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//     searchTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
//     #endif

//     float r = child[maxi]->searchandPropagate(evaluator);
//     W.fetch_add(r + virtualLoss);
//     return -r * decay;
// }

// std::pair<int, int> Node::selectMove(float temp){
//     if(winmove.first >= 0)
//         return winmove;
//     if(legal.size() == 1 && game->scoreWinner() != game->getTurn()){ // if lost, resign
//         return {-1, -1};
//     }

//     std::vector<float> weights(legal.size());
//     std::vector<float> cumulative(legal.size());

//     int maxi, maxn = -1, index;
//     for(int i=0; i<legal.size(); ++i){
//         if(child[i]->N > maxn){
//             maxn = child[i]->N;
//             maxi = i;
//         }
//         weights[i] = std::pow(child[i]->N, temp);
//     }

//     std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

//     if(temp < 5.0f){
//         std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
//         float rnd = dist(gen);

//         auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
//         index = std::distance(cumulative.begin(), it);
//         return legal[index];
//     }

//     for(int i=0; i<legal.size(); ++i){
//         std::cout << "move : " << legal[i].first << " " << legal[i].second << " sc: " << child[i]->N << " wc: " << 
//         child[i]->W << " initQ : " << child[i]->initQ << " P " << child[i]->P << std::endl;
//     }
//     return legal[maxi];
// }

// MoveData Node::selectMoveProb(float temp){
//     std::array<float, outputSize> visitPortion;
//     std::vector<float> cumulative(legal.size()), weights(legal.size());

//     visitPortion.fill(0.0f);
//     if(winmove.first >= 0)
//         return {winmove, visitPortion};
//     if(legal.size() == 1 && game->scoreWinner() != game->getTurn()){ // if lost, resign
//         return {{-1, -1}, visitPortion};
//     }

//     int maxi, maxn = -1;
//     for(int i=0; i<legal.size(); ++i){
//         if(child[i]->N > maxn){
//             maxn = child[i]->N;
//             maxi = i;
//         }
//         weights[i] = std::pow(child[i]->N, temp);
//         visitPortion[legal[i].first * colSize + legal[i].second] = static_cast<float>(child[i]->N)/N;
//     }

//     // std::cout << "visit portion" << std::endl;
//     // for(int i=0; i<outputSize; ++i)
//     //     std::cout << visitPortion[i] << " ";
//     // std::cout << std::endl;

//     if(temp < 5.0f){
//         std::partial_sum(weights.begin(), weights.end(), cumulative.begin());

//         std::uniform_real_distribution<float> dist(0.0f, cumulative.back());
//         float rnd = dist(gen);

//         auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rnd);
//         size_t index = std::distance(cumulative.begin(), it);

//         // std::cout << "make move : " << legal[index].first << " " << legal[index].second << " win count : " << child[index]->W << " visit count : " << child[index]->N <<
//         // " prob : " << child[index]->P << " eval : " << child[index]->initQ << "\n";

//         #ifdef measureTime
//         std::cout << expandTime << " " << expandWaitTime << " " << evaluateTime << " " << searchTime << " " << searchWaitTime << " " << copyTime << " " << makeMoveTime << " " << extraTime << "\n";
//         expandTime = expandWaitTime = evaluateTime = searchTime = searchWaitTime = copyTime = makeMoveTime = extraTime = 0;
//         #endif

//         return {legal[index], visitPortion};
//     }


//     // std::cout << "make move : " << legal[maxi].first << " " << legal[maxi].second << " win count : " << child[maxi]->W << " visit count : " << child[maxi]->N << 
//     // " prob : " << child[maxi]->P << " eval : " << child[maxi]->initQ << "\n";

//     #ifdef measureTime
//     std::cout << expandTime << " " << expandWaitTime << " " 
//     << evaluateTime << " " << searchTime << " " << searchWaitTime <<  " " << copyTime << " " << makeMoveTime << " " << extraTime << "\n";
//     expandTime = expandWaitTime = evaluateTime = searchTime = searchWaitTime = copyTime = makeMoveTime = extraTime = 0;
//     #endif

//     return {legal[maxi], visitPortion};
// }

// Node* Node::jump(std::pair<int, int> move){
//     if(N == 0){
//         expand();
//         N++;
//     }

//     int idx = -1;
//     Node* next;

//     //std::cout << "# of possible moves " << legal.size() << std::endl;
//     for(int i=0; i<legal.size(); ++i){
//         if(legal[i] == move){
//             idx = i;
//             next = child[idx];
//         }

//         else{
//             //std::cout << i << " ";
//             child[i]->deletetree();
//             delete child[i];
//         }
//     }

//     child.clear();
//     return next;
// }

// void MCTS::runSimulation(){
//     std::vector<std::thread> threads;
//     int base_simulations = playout / searchThreadN;
//     int extra_simulations = playout % searchThreadN; // Remaining simulations

//     // Launch multiple threads
//     for (int i = 0; i < searchThreadN; ++i) {
//         int num_runs = base_simulations + (i < extra_simulations ? 1 : 0); // Distribute extra work
//         searchPool.enqueue([this, num_runs]{
//             evaluator.activeSearchThread.fetch_add(1);

//             for(int j=0; j<num_runs; ++j)
//                 root->searchandPropagate(evaluator);

//             evaluator.activeSearchThread.fetch_add(-1);
//         });
//     }

//     searchPool.waitForAll();
// }

// std::pair<int, int> MCTS::getMove(float temp){
//     runSimulation();
//     return root->selectMove(temp);
// }

// MoveData MCTS::getMoveProb(float temp){
//     runSimulation();
//     return root->selectMoveProb(temp);
// }

// void MCTS::jump(std::pair<int, int> move){
//     Node* nroot = root->jump(move);
//     delete root;
//     root = nroot;
// }

// void MCTS::reset(){
//     root->deletetree();
//     delete(root);
//     root = new Node(new Game());
// }

// MCTS::MCTS(int playout, PolicyValueNet& net) : net(net), playout(playout), root(new Node(new Game())), evaluator(net), searchPool(searchThreadN){}


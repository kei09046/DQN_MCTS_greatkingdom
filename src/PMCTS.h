#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"
#include "evaluator.h"
#include "gamerules.h"
#include <optional>
#include <vector>
#include <utility>
#include <unordered_map>

class Node;
class Analysis;
using TransTable = std::unordered_map<HashValue, std::pair<Node*, unsigned int>>;

class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, TransTable* const transposTable);

    Move selectMove(float temperature, int minVisit);

    // return move + training target.
    MoveData selectMoveWithStat(float temperature, int minVisit);

    Node* jump(Move move);

    void addDirichletNoise(Evaluator* evaluator);

    void deleteTree();

    void deleteTree(Node* exception);

    inline const Game& game_() const{
        return game;
    }

private:
    friend class MCTS;
    friend class Analysis;

    Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, initQ, S, Wp; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    float minN; // # of minimum-visit playouts counted in N but not backed up into W/S/Wp (only ever nonzero on root)
    // # of visits W/S/Wp are summed over. Divide by this, not N, to get this node's own Q.
    inline float valueN() const { return N - minN; }
    // S : mean score difference Wp : mean win probability
    std::vector<float> edgeP, edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.
    const Color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation
    std::vector<Node*> child;

    int forcedState; // if 0 : not forced win/loss +k : win in k move -k : lose in k move
    bool expanded; // check if node has been expanded.
    
    TransTable* const transposTable;

    void addChild(const Move& move, int idx = -1); // add child to the node. Will be added at the end by default.

    void expand();

    int selectChildInSearch(float fpu);
};


class alignas(64) MCTS{
public:
    MCTS(Evaluator* evaluator, SearchParams params = globalConfig.searchParams());
    ~MCTS();
    MCTS(MCTS&& other) noexcept;

    // noise : add Dirichlet noise to the root (if enabled in the config).
    void runSimulation(const int playMode, const int nPlayout, const int timeLimit, const int minSearchPerChild = -1, const bool fastSearch = false,
    const bool earlyQuit = false);

    Move getMove(float temperature);

    MoveData getMoveWithStat(float temperature);

    // fast search of playout cap randomization : fastPlayout more playouts, without Dirichlet noise or forced visits.
    // Only moves the game forward; its result isn't a training target.
    Move getFastMove(float temperature);

    // pick a move from the current root's search result (no additional search).
    Move selectMove(float temperature);

    // getEval returns evaluation from player's perspective, which is opposite from internally stored value.
    std::pair<float, float> getEval();

    bool jump(Move move);

    // resets MCTS tree and set root to startPosition.
    void reset(const Game& startPos);

    // exposes the root node's internal game state, for debugging desyncs against an external mirror (e.g. TrainPipeline::game_manager).
    inline const Game& currentGame() const{
        return root->game_();
    }

    inline int rootForcedState() const{
        return root->forcedState;
    }

    inline int rootVisits() const{
        return static_cast<int>(root->N);
    }

    inline const SearchParams& searchParams() const{
        return params;
    }

    // Associates this MCTS instance with the Analysis object that owns debug-mode state (see
    // Analysis::debugMode in analysis.hpp) -- purely so playout()/updateEval() below can reach
    // it for the debug-line streaming they do mid-search (see printPlayoutDebugLine's own
    // comment). Never called for any MCTS instance training touches, so `analysis` stays
    // nullptr there and the hot-path checks below cost nothing more than the old inline bool
    // did. Non-owning: whoever constructs the Analysis (ModelCompare::analyze) still owns it.
    inline void attachAnalysis(Analysis* a){
        analysis = a;
    }

    #ifdef measureTime
    std::vector<int> getTimeStats() const;

    void resetTimeStats();
    #endif

private:
    friend class Analysis;

    Node* root;
    Evaluator* evaluator; // shared along multiple MCTS instances
    TransTable* transposTable;
    SearchParams params; // fpu / minVisitRatio used by this instance.

    // Non-owning; null unless attachAnalysis() was called (see above). Only playout()/updateEval()
    // below read this, to stream debug lines through it mid-search -- everything else Analysis
    // does (printAnalysis, printVariation, ...) is driven the other way, by a caller handing its
    // own MCTS instance into an Analysis method, so doesn't need MCTS to hold this at all.
    Analysis* analysis = nullptr;
    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation,
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int timeLimit, int forcedFirstSearch = -1);

    void updateEval(const std::shared_ptr<NNResultBuf> buf, const std::vector<Node*> path, Node* cur);

    std::vector<Node*> debugPath(const std::vector<Node*>& path) const;

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW, float evalS);
};
#endif